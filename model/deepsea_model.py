import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from einops import rearrange
from itertools import accumulate

from .layer import (
    PatchEmbed,
    PatchMerging,
    SwinEncoderStage,
    PatchExpanding,
    SwinDecoderStage,
)

from .utils import (
    compute_land_mask,
    prepare_land_mask_2d,
)

from .base import ORCADLOutput, ORCADLConfig, ORCADLPreTrainedModel


class EncoderModule(nn.Module):
    def __init__(self, config, in_chans, use_mask_token=False, is_atmo=False):
        super().__init__()

        self.num_stages = len(config.enc_depths)
        self.embed_dim = config.embed_dim
        self.window_size = config.window_size
        self.patch_size = config.patch_size

        self.patch_embed = PatchEmbed(
            config.input_shape, config.patch_size, in_chans,
            config.embed_dim, config.patch_norm, config.use_absolute_embeddings, use_mask_token
        )

        self.max_t = config.max_t

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.enc_depths))]

        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            stage = SwinEncoderStage(
                config,
                self.window_size,
                dim=int(config.embed_dim * 2**i_stage),
                depth=config.enc_depths[i_stage],
                num_heads=config.enc_heads[i_stage],
                drop_path=dpr[sum(config.enc_depths[:i_stage]):sum(config.enc_depths[:i_stage + 1])],
                downsample=PatchMerging if i_stage < self.num_stages - 1 else None,
                is_atmo=is_atmo,
            )
            self.stages.append(stage)

        self.num_features = int(config.embed_dim * 2**(self.num_stages - 1))

        self.norm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)

    def forward(self, x, lead_time, all_land_mask_pad, all_land_mask_pad_shifted, mask=None):
        all_hidden_states_before_downsample = ()

        x = self.patch_embed(x, mask)
        x = self.pos_drop(x)
        x = rearrange(x, 'b c h w -> b h w c').contiguous()

        for i in range(self.num_stages):
            x, x_before_downsample = self.stages[i](
                x,
                all_land_mask_pad[i],
                all_land_mask_pad_shifted[i],
                lead_time,
            )
            all_hidden_states_before_downsample += (x_before_downsample,)

        x = self.norm(x)

        return x, all_hidden_states_before_downsample


class DecoderModule(nn.Module):
    def __init__(self, config, out_chans):
        super().__init__()

        self.num_stages = len(config.enc_depths)
        self.embed_dim = config.embed_dim
        self.window_size = config.window_size
        self.patch_size = config.patch_size

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.enc_depths))]

        self.stages = nn.ModuleList()
        self.concat_proj_layers = nn.ModuleList()
        for i_stage in range(self.num_stages - 1, -1, -1):
            stage = SwinDecoderStage(
                config,
                self.window_size,
                dim=int(config.embed_dim * 2**i_stage),
                depth=config.enc_depths[i_stage],
                num_heads=config.enc_heads[i_stage],
                drop_path=dpr[sum(config.enc_depths[:i_stage]):sum(config.enc_depths[:i_stage + 1])],
                upsample=PatchExpanding if i_stage > 0 else None,
            )
            self.stages.append(stage)
            if i_stage < self.num_stages - 1:
                self.concat_proj_layers.append(
                    nn.Linear(config.embed_dim * 2**i_stage * 2, config.embed_dim * 2**i_stage)
                )

        self.num_features = int(config.embed_dim)

        self.norm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.max_t = config.max_t

        self.final_proj = nn.ConvTranspose2d(
            config.embed_dim, out_chans, config.patch_size, config.patch_size
        )

    def forward(self, x, lead_time, enc_x, all_land_mask_pad, all_land_mask_pad_shifted):
        for i in range(self.num_stages):
            if i > 0:
                x = torch.cat([x, enc_x[self.num_stages - 1 - i]], dim=-1)
                x = self.concat_proj_layers[i - 1](x)

            x = self.stages[i](
                x,
                all_land_mask_pad[self.num_stages - 1 - i],
                all_land_mask_pad_shifted[self.num_stages - 1 - i],
                lead_time,
            )

        x = self.norm(x)
        x = rearrange(x, 'n h w c -> n c h w').contiguous()

        return self.final_proj(x)


class OceanEncoders(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.in_chans = config.in_chans
        self.num_enc = len(config.in_chans)
        self.encoder_list = nn.ModuleList()
        for in_chans in config.in_chans:
            self.encoder_list.append(EncoderModule(config, in_chans))
        self.proj = nn.Linear(
            config.embed_dim * 2**(len(config.enc_depths) - 1) * len(config.in_chans),
            config.lg_hidden_dim,
        )

        self.max_t = config.max_t

    def forward(self, x, lead_time, all_land_mask_pad, all_land_mask_pad_shifted, mask=None):
        x_split = torch.split(x, self.in_chans, dim=1)
        all_last_x, all_hidden_x = [], []
        for i in range(self.num_enc):
            x, hidden_x = self.encoder_list[i](
                x_split[i], lead_time, all_land_mask_pad, all_land_mask_pad_shifted
            )
            all_last_x.append(x)
            all_hidden_x.append(hidden_x)
        out = self.proj(torch.cat(all_last_x, dim=-1))

        return out, all_hidden_x


class OceanDecoders(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.out_chans = config.out_chans
        self.num_dec = len(config.out_chans)
        self.decoder_list = nn.ModuleList()
        for out_chans in config.out_chans:
            self.decoder_list.append(DecoderModule(config, out_chans))
        self.proj = nn.Linear(
            config.lg_hidden_dim,
            config.embed_dim * 2**(len(config.enc_depths) - 1) * len(config.in_chans),
        )
        self.split_dims = config.embed_dim * 2**(len(config.enc_depths) - 1)

        self.max_t = config.max_t

    def forward(self, x, lead_time, all_enc_x, all_land_mask_pad, all_land_mask_pad_shifted):
        x = self.proj(x)

        x_split = torch.split(x, self.split_dims, dim=-1)
        all_x = []
        for i in range(self.num_dec):
            x = self.decoder_list[i](
                x_split[i], lead_time, all_enc_x[i], all_land_mask_pad, all_land_mask_pad_shifted
            )
            all_x.append(x)
        out = torch.cat(all_x, dim=1)

        return out


class FusionModule(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_stages = len(config.lg_depths)
        self.hidden_size = config.lg_hidden_dim

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.lg_depths))]

        window_size = (
            config.input_shape[0] // config.patch_size[0] // (2 ** (len(config.enc_depths) - 1)),
            config.input_shape[1] // config.patch_size[1] // (2 ** (len(config.enc_depths) - 1)),
        )
        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            stage = SwinEncoderStage(
                config,
                window_size,
                dim=self.hidden_size,
                depth=config.lg_depths[i_stage],
                num_heads=config.lg_heads[i_stage],
                drop_path=dpr[sum(config.lg_depths[:i_stage]):sum(config.lg_depths[:i_stage + 1])],
                downsample=None,
            )
            self.stages.append(stage)

        self.pos_drop = nn.Dropout(p=config.drop_rate)
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size[0] * window_size[1], self.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, lead_time, land_mask_pad, land_mask_pad_shifted):
        B, H, W, C = x.shape

        x = x.reshape(B, -1, C)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.reshape(B, H, W, C)

        for i in range(self.num_stages):
            x, _ = self.stages[i](x, land_mask_pad, land_mask_pad_shifted, lead_time)
        return x


class AtmoEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.encoder = EncoderModule(
            config, config.atmo_dims, use_mask_token=config.use_mask_token, is_atmo=True
        )
        self.proj = nn.Linear(
            config.embed_dim * 2**(len(config.enc_depths) - 1),
            config.lg_hidden_dim,
        )

    def forward(self, x, lead_time, all_land_mask_pad, all_land_mask_pad_shifted, mask=None):
        x, _ = self.encoder(x, lead_time, all_land_mask_pad, all_land_mask_pad_shifted, mask)
        x = self.proj(x)
        return x


class ORCADLModel(ORCADLPreTrainedModel):
    def __init__(self, config: ORCADLConfig) -> None:
        super().__init__(config)

        self.enc_ocean = OceanEncoders(config)
        self.fusion = FusionModule(config)
        self.dec_ocean = OceanDecoders(config)
        self.loss_type = config.loss_type

        self.use_land_mask = config.use_land_mask

        land_mask = compute_land_mask(tuple(config.lat_space), tuple(config.lon_space))
        self.register_buffer("land_mask", land_mask)

        H, W = tuple(config.input_shape)

        window_size_mix = (
            H // config.patch_size[0] // (2 ** (len(config.enc_depths) - 1)),
            W // config.patch_size[1] // (2 ** (len(config.enc_depths) - 1)),
        )

        all_land_mask_pad, all_land_mask_pad_shifted = prepare_land_mask_2d(
            land_mask, config.patch_size, config.window_size, H, W, len(config.enc_depths)
        )
        self.all_land_mask_pad = all_land_mask_pad
        self.all_land_mask_pad_shifted = all_land_mask_pad_shifted

        all_land_mask_pad_mix, all_land_mask_pad_shifted_mix = prepare_land_mask_2d(
            land_mask, config.patch_size, window_size_mix, H, W, len(config.enc_depths)
        )
        self.land_mask_pad_mix = all_land_mask_pad_mix[-1]
        self.land_mask_pad_shifted_mix = all_land_mask_pad_shifted_mix[-1]

        self.in_chans = config.in_chans
        self.out_chans = config.out_chans

        self.split_chans = list(accumulate(self.out_chans))

        self.max_t = config.max_t
        self.in_steps = config.in_steps

        self.post_init()

    def compute_loss(
        self,
        preds: torch.FloatTensor,
        labels: torch.FloatTensor,
    ):
        if labels is None:
            return None

        num_dims = len(preds.shape)
        if num_dims == 4:
            preds = preds[:, :, self.land_mask]
            labels = labels[:, :, self.land_mask]
        elif num_dims == 5:
            preds = preds[:, :, :, self.land_mask]
            labels = labels[:, :, :, self.land_mask]
        else:
            assert 0

        if self.loss_type == 'mae':
            loss = F.l1_loss(preds, labels)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(preds, labels)
        elif self.loss_type == 'rmse':
            loss = torch.sqrt(torch.mean((preds - labels)**2))
        elif self.loss_type == 'rmse_new':
            loss = torch.sqrt(torch.mean((preds - labels)**2, dim=(-1, -2))).mean()
        elif self.loss_type == 'balance_rmse':
            loss = torch.sqrt(torch.mean((preds - labels)**2, dim=(0, -1)))

            if num_dims == 4:
                tos_loss = loss[32]
                loss = loss / (loss / tos_loss).detach()
            else:
                tos_loss = loss[:, 32:33]
                loss = loss / (loss / tos_loss).detach()

            loss = loss.mean()

        return loss

    def forward_single_step(
        self,
        ocean_vars: torch.FloatTensor,
        lead_time: torch.LongTensor,
        mask: torch.FloatTensor = None,
        labels: torch.FloatTensor = None,
        return_dict: bool = None,
    ):
        x, enc_x = self.enc_ocean(
            ocean_vars, lead_time, self.all_land_mask_pad, self.all_land_mask_pad_shifted, mask
        )
        x = self.fusion(x, lead_time, self.land_mask_pad_mix, self.land_mask_pad_shifted_mix)
        logits = self.dec_ocean(
            x, lead_time, enc_x, self.all_land_mask_pad, self.all_land_mask_pad_shifted
        )

        loss = self.compute_loss(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return ORCADLOutput(
            loss=loss,
            preds=logits,
        )

    def forward_multi_steps(
        self,
        ocean_vars: torch.FloatTensor,
        mask: torch.FloatTensor = None,
        labels: torch.FloatTensor = None,
        predict_time_steps: int = None,
        return_dict: bool = None,
    ):
        B, _, H, W = ocean_vars.shape
        out_chans = sum(self.config.out_chans)
        all_preds = torch.zeros(B, predict_time_steps, out_chans, H, W, device=ocean_vars.device)

        for t in range(predict_time_steps):
            lead_time = torch.tensor(t, device=ocean_vars.device).repeat(B)
            preds = self.forward_single_step(
                ocean_vars=ocean_vars,
                lead_time=lead_time,
                mask=mask,
                return_dict=False,
            )[0]

            all_preds[:, t] = preds

            if (t + 1) % self.max_t == 0:
                data = []
                for i in range(len(self.split_chans)):
                    if i == 0:
                        slic = slice(0, self.split_chans[i])
                    else:
                        slic = slice(self.split_chans[i - 1], self.split_chans[i])
                    data.extend([all_preds[:, t - j, slic] for j in range(self.in_steps - 1, -1, -1)])

                ocean_vars = torch.cat(data, dim=1)

        loss = self.compute_loss(all_preds, labels)

        if not return_dict:
            output = (all_preds,)
            return ((loss,) + output) if loss is not None else output

        return ORCADLOutput(
            loss=loss,
            preds=all_preds,
        )

    def forward(
        self,
        ocean_vars: torch.FloatTensor,
        lead_time: Optional[torch.LongTensor] = None,
        mask: torch.FloatTensor = None,
        labels: Optional[torch.FloatTensor] = None,
        predict_time_steps: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ):
        if return_dict is None:
            return_dict = self.config.use_return_dict
        if predict_time_steps is None:
            predict_time_steps = getattr(self.config, 'predict_time_steps', 1)

        if predict_time_steps == 1:
            if lead_time is None:
                lead_time = torch.zeros(ocean_vars.shape[0], device=ocean_vars.device).long()
            return self.forward_single_step(
                ocean_vars=ocean_vars,
                lead_time=lead_time,
                mask=mask,
                labels=labels,
                return_dict=return_dict,
            )
        return self.forward_multi_steps(
            ocean_vars=ocean_vars,
            mask=mask,
            labels=labels,
            predict_time_steps=predict_time_steps,
            return_dict=return_dict,
        )
