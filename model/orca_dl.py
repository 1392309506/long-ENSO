import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from einops import rearrange
from itertools import accumulate
from dataclasses import dataclass
from timm.models.layers import trunc_normal_
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import ModelOutput, PreTrainedModel

from .layer import (
    PatchEmbed,
    PatchMerging,
    SwinEncoderStage,
    PatchExpanding,
    SwinDecoderStage,
    SwinLayer,
    RotaryTimeEmbed, DiffusionTimeEmbedding,
)

from .utils import (
    compute_land_mask,
    prepare_land_mask_2d,
)

def _timestep_embedding(timesteps: torch.LongTensor, dim: int) -> torch.FloatTensor:
    half = dim // 2
    denom = max(half - 1, 1)
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / denom
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class PerturbationModel(nn.Module):
    def __init__(self, in_chans: int, cond_chans: int, base_chans: int, time_embed_dim: int):
        super().__init__()
        self.time_embed = DiffusionTimeEmbedding(time_embed_dim, base_chans)
        self.in_proj = nn.Conv2d(in_chans + cond_chans, base_chans, kernel_size=3, padding=1)
        self.res1 = _ResBlock(base_chans)
        self.res2 = _ResBlock(base_chans)
        self.out_proj = nn.Conv2d(base_chans, in_chans, kernel_size=3, padding=1)

    def forward(self, x_t, t, cond):
        # x_t: (B, C, H, W), cond: (B, C_cond, H, W)
        h = torch.cat([x_t, cond], dim=1)
        h = self.in_proj(h)
        t_emb = self.time_embed(_timestep_embedding(t, self.time_embed.mlp[0].in_features))
        h = h + t_emb[:, :, None, None]
        h = self.res1(h)
        h = self.res2(h)
        return self.out_proj(h)

@dataclass
class ORCADLOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    preds: torch.FloatTensor = None


class ORCADLConfig(PretrainedConfig):
    def __init__(
        self,
        lat_space=(-63.5, 63.5, 128),
        lon_space=(0.5, 359.5, 360),
        use_land_mask=True,
        patch_size=(2, 3),
        in_chans=[16, 16, 1, 16, 16, 1],
        out_chans=[16, 16, 1, 16, 16, 1],
        embed_dim=96,
        lg_hidden_dim=1152,
        enc_depths=(2, 2, 2),
        enc_heads=(3, 6, 12),
        lg_depths=(2, 2),
        lg_heads=(12, 12),
        window_size=(8, 15),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        use_absolute_embeddings=True,
        var_list=[],
        var_index=[],
        loss_type='rmse',
        max_t=None,
        atmo_dims=3,
        atmo_embed_dims=64,
        diffusion_steps=1000,
        diffusion_beta_start=1e-4,
        diffusion_beta_end=2e-2,
        diffusion_time_embed_dim=256,
        mask_patch_size=(8,12),
        mask_ratio=0.8,
        use_mask_token=False,
        is_moe=True,
        is_moe_encoder=True,
        is_moe_decoder=True,
        is_moe_atmo=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lat_space = lat_space
        self.lon_space = lon_space
        self.use_land_mask = use_land_mask
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.enc_depths = enc_depths
        self.enc_heads = enc_heads
        self.lg_depths = lg_depths
        self.lg_heads = lg_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop = attn_drop
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.use_absolute_embeddings = use_absolute_embeddings

        self.var_list = var_list
        self.var_index = var_index

        # this indicates the channel dimension after the last stage of the model
        self.lg_hidden_dim = lg_hidden_dim
        self.input_shape = (self.lat_space[-1], self.lon_space[-1])
        self.loss_type = loss_type
        self.max_t = max_t

        self.atmo_dims = atmo_dims
        self.atmo_embed_dims = atmo_embed_dims
        self.diffusion_steps = diffusion_steps
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        self.diffusion_time_embed_dim = diffusion_time_embed_dim

        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio

        self.use_mask_token = use_mask_token
        self.is_moe = is_moe
        self.is_moe_encoder = is_moe_encoder
        self.is_moe_decoder = is_moe_decoder
        self.is_moe_atmo = is_moe_atmo

        self.in_steps = in_chans[0] // out_chans[0]

    def update(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_args(self, args):
        config_dict = self.to_diff_dict()
        args_dict = args.to_dict()
        update_config = {}
        for key in config_dict.keys():
            if key in args_dict and args_dict[key] is not None:
                update_config[key] = args_dict[key]
        self.update(update_config)


class EncoderModule(nn.Module):
    def __init__(self, config, in_chans, use_mask_token=False, is_atmo=False):
        super().__init__()

        self.num_stages = len(config.enc_depths)
        self.embed_dim = config.embed_dim
        self.window_size = config.window_size
        self.patch_size = config.patch_size

        self.patch_embed = PatchEmbed(
            config.input_shape, config.patch_size, in_chans,
            config.embed_dim, config.patch_norm, config.use_absolute_embeddings, use_mask_token)

        self.max_t = config.max_t

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.enc_depths))]

        # build stages
        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            stage = SwinEncoderStage(
                config,
                self.window_size,
                dim=int(config.embed_dim * 2**i_stage),
                depth=config.enc_depths[i_stage],
                num_heads=config.enc_heads[i_stage],
                drop_path=dpr[sum(config.enc_depths[:i_stage]):sum(config.enc_depths[:i_stage + 1])],
                downsample=PatchMerging if i_stage < self.num_stages-1 else None,
                is_atmo=is_atmo)
            self.stages.append(stage)

        self.num_features = int(config.embed_dim * 2**(self.num_stages-1))

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
                lead_time
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

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate,
                                                sum(config.enc_depths))]  # stochastic depth decay rule

        # build layers
        self.stages = nn.ModuleList()
        self.concat_proj_layers = nn.ModuleList()
        for i_stage in range(self.num_stages-1, -1, -1):
            stage = SwinDecoderStage(
                config,
                self.window_size,
                dim=int(config.embed_dim * 2**i_stage),
                depth=config.enc_depths[i_stage],
                num_heads=config.enc_heads[i_stage],
                drop_path=dpr[sum(config.enc_depths[:i_stage]):sum(config.enc_depths[:i_stage + 1])],
                upsample=PatchExpanding if i_stage > 0 else None)
            self.stages.append(stage)
            if i_stage < self.num_stages-1:
                self.concat_proj_layers.append(
                    nn.Linear(config.embed_dim * 2**i_stage * 2, config.embed_dim * 2**i_stage)
                )

        self.num_features = int(config.embed_dim)

        # add a norm layer for each output
        self.norm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        
        self.max_t = config.max_t

        self.final_proj = nn.ConvTranspose2d(
                config.embed_dim, out_chans, config.patch_size, config.patch_size)

    def forward(self, x, lead_time, enc_x, all_land_mask_pad, all_land_mask_pad_shifted):

        for i in range(self.num_stages):
            if i > 0:
                x = torch.cat([x, enc_x[self.num_stages-1-i]], dim=-1)
                x = self.concat_proj_layers[i-1](x)

            x = self.stages[i](
                x,
                all_land_mask_pad[self.num_stages-1-i],
                all_land_mask_pad_shifted[self.num_stages-1-i],
                lead_time
            )

        # x = rearrange(x, 'n c d h w -> n d h w c')
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
        self.proj = nn.Linear(config.embed_dim * 2**(len(config.enc_depths) - 1) * len(config.in_chans), config.lg_hidden_dim)

        self.max_t = config.max_t

    def forward(self, x, lead_time, all_land_mask_pad, all_land_mask_pad_shifted, mask=None):
        
        x_split = torch.split(x, self.in_chans, dim=1)
        all_last_x, all_hidden_x = [], []
        for i in range(self.num_enc):
            x, hidden_x = self.encoder_list[i](x_split[i], lead_time, all_land_mask_pad, all_land_mask_pad_shifted)
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
        self.proj = nn.Linear(config.lg_hidden_dim, config.embed_dim * 2**(len(config.enc_depths) - 1) * len(config.in_chans))
        self.split_dims = config.embed_dim * 2**(len(config.enc_depths) - 1)

        self.max_t = config.max_t


    def forward(self, x, lead_time, all_enc_x, all_land_mask_pad, all_land_mask_pad_shifted):

        x = self.proj(x)

        x_split = torch.split(x, self.split_dims, dim=-1)
        all_x = []
        for i in range(self.num_dec):
            x = self.decoder_list[i](x_split[i], lead_time, all_enc_x[i], all_land_mask_pad, all_land_mask_pad_shifted)
            all_x.append(x)
        out = torch.cat(all_x, dim=1)

        return out


class FusionModule(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_stages = len(config.lg_depths)
        self.hidden_size = config.lg_hidden_dim

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.lg_depths))]

        window_size = (config.input_shape[0] // config.patch_size[0] // (2 ** (len(config.enc_depths) -1)),
                       config.input_shape[1] // config.patch_size[1] // (2 ** (len(config.enc_depths) -1)))
        # build stages
        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            stage = SwinEncoderStage(
                config,
                window_size, # if i_stage == 0 else config.window_size,
                dim=self.hidden_size,
                depth=config.lg_depths[i_stage],
                num_heads=config.lg_heads[i_stage],
                drop_path=dpr[sum(config.lg_depths[:i_stage]):sum(config.lg_depths[:i_stage + 1])],
                downsample=None)
            self.stages.append(stage)

        # self.norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size[0] * window_size[1], self.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        # 移除 self.rotate_atmo = RotaryTimeEmbed(...)
        # self.rotate_atmo = RotaryTimeEmbed(self.hidden_size)

    def forward(self, x, lead_time, land_mask_pad, land_mask_pad_shifted):
        # 移除了 atmo_x 参数
        B, H, W, C = x.shape

        # 移除所有 atmo_x 的加权融合逻辑
        # 直接对合并后的海洋特征进行全局耦合
        # atmo_x = atmo_x.permute(0, 3, 1, 2).contiguous()
        # atmo_x = self.rotate_atmo(atmo_x, lead_time if atmo_lead_time is None else atmo_lead_time)
        # atmo_x = atmo_x.permute(0, 2, 3, 1).contiguous()

        # atmo_w = torch.sum(atmo_x * x, dim=-1, keepdim=True)

        # x = x + atmo_w * atmo_x

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

        self.encoder = EncoderModule(config, config.atmo_dims, use_mask_token=config.use_mask_token, is_atmo=True)
        self.proj = nn.Linear(config.embed_dim * 2**(len(config.enc_depths) - 1), config.lg_hidden_dim)

    def forward(self, x, lead_time, all_land_mask_pad, all_land_mask_pad_shifted, mask=None):
        x, _ = self.encoder(x, lead_time, all_land_mask_pad, all_land_mask_pad_shifted, mask)
        x = self.proj(x)
        return x


class ORCADLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ORCADLConfig
    main_input_name = "ocean_vars"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, SwinLayer):
            module.gradient_checkpointing = value


class ORCADLModel(ORCADLPreTrainedModel):
    """
    The ORCA-DL model class.
    """
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

        window_size_mix = (H // config.patch_size[0] // (2 ** (len(config.enc_depths) -1)), 
                       W // config.patch_size[1] // (2 ** (len(config.enc_depths) -1)))

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
            loss = torch.sqrt(torch.mean((preds-labels)**2))
        elif self.loss_type == 'rmse_new':
            loss = torch.sqrt(torch.mean((preds-labels)**2, dim=(-1,-2))).mean()
        elif self.loss_type == 'balance_rmse':
            loss = torch.sqrt(torch.mean((preds-labels)**2, dim=(0, -1)))

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
        return_dict: bool = None
    ):

        x, enc_x = self.enc_ocean(ocean_vars, lead_time, self.all_land_mask_pad, self.all_land_mask_pad_shifted, mask)
        x = self.fusion(x, lead_time, self.land_mask_pad_mix, self.land_mask_pad_shifted_mix)
        logits = self.dec_ocean(x, lead_time, enc_x, self.all_land_mask_pad, self.all_land_mask_pad_shifted)

        loss = self.compute_loss(logits, labels)

        if not return_dict:
            output = (logits, )
            return ((loss,) + output) if loss is not None else output

        return ORCADLOutput(
            loss=loss,
            preds=logits
        )

    def forward_multi_steps(
        self,
        ocean_vars: torch.FloatTensor,
        mask: torch.FloatTensor = None,
        labels: torch.FloatTensor = None,
        predict_time_steps: int = None,
        return_dict: bool = None
    ):

        B, _, H, W = ocean_vars.shape
        out_chans = sum(self.config.out_chans)
        all_preds = torch.zeros(B, predict_time_steps, out_chans, H, W, device=ocean_vars.device)
        
        for t in range(predict_time_steps):
            lead_time = torch.tensor(t, device=ocean_vars.device).repeat(B)
            preds = self.forward_single_step(ocean_vars=ocean_vars,
                                             lead_time=lead_time,
                                             mask=mask,
                                             return_dict=False)[0]

            all_preds[:, t] = preds

            if (t+1) % self.max_t == 0:
                data = []
                for i in range(len(self.split_chans)):
                    if i == 0:
                        slic = slice(0, self.split_chans[i])
                    else:
                        slic = slice(self.split_chans[i-1], self.split_chans[i])
                    data.extend([all_preds[:, t-j, slic] for j in range(self.in_steps-1, -1, -1)])

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
        return_dict: Optional[bool] = None
    ):
        if return_dict is None:
            return_dict = self.config.use_return_dict
        if predict_time_steps is None:
            predict_time_steps = getattr(self.config, 'predict_time_steps', 1)

        if predict_time_steps == 1:
            if lead_time is None:
                lead_time = torch.zeros(ocean_vars.shape[0], device=ocean_vars.device).long()
            return self.forward_single_step(ocean_vars=ocean_vars,
                                            lead_time=lead_time,
                                            mask=mask,
                                            labels=labels,
                                            return_dict=return_dict)
        return self.forward_multi_steps(ocean_vars=ocean_vars,
                                        mask=mask,
                                        labels=labels,
                                        predict_time_steps=predict_time_steps,
                                        return_dict=return_dict)


class ORCADLPerturbationModel(ORCADLPreTrainedModel):
    def __init__(self, config: ORCADLConfig, base_model: Optional[ORCADLModel] = None, freeze_base_model: bool = True):
        super().__init__(config)
        self.base_model = base_model if base_model is not None else ORCADLModel(config)
        if freeze_base_model:
            for p in self.base_model.parameters():
                p.requires_grad = False

        out_chans = sum(config.out_chans)
        cond_chans = out_chans + config.atmo_dims
        self.noise_model = PerturbationModel(
            in_chans=out_chans,
            cond_chans=cond_chans,
            base_chans=config.embed_dim,
            time_embed_dim=config.diffusion_time_embed_dim,
        )

        betas = torch.linspace(config.diffusion_beta_start, config.diffusion_beta_end, config.diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _p_mean_variance(self, x_t: torch.FloatTensor, t: torch.LongTensor, cond: torch.FloatTensor):
        eps_pred = self.noise_model(x_t, t, cond)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus) * eps_pred)
        return model_mean, beta_t

    def sample_delta(self, cond: torch.FloatTensor, shape: torch.Size):
        x_t = torch.randn(shape, device=cond.device, dtype=cond.dtype)
        for i in reversed(range(self.config.diffusion_steps)):
            t = torch.full((shape[0],), i, device=cond.device, dtype=torch.long)
            model_mean, beta_t = self._p_mean_variance(x_t, t, cond)
            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = model_mean + torch.sqrt(beta_t) * noise
            else:
                x_t = model_mean
        return x_t

    def forward(
        self,
        ocean_vars: torch.FloatTensor,
        atmo_vars: torch.FloatTensor,
        lead_time: Optional[torch.LongTensor] = None,
        atmo_lead_time: Optional[torch.LongTensor] = None,
        mask: torch.FloatTensor = None,
        labels: Optional[torch.FloatTensor] = None,
        predict_time_steps: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ):
        if return_dict is None:
            return_dict = self.config.use_return_dict

        base_out = self.base_model(
            ocean_vars=ocean_vars,
            lead_time=lead_time,
            mask=mask,
            labels=None,
            predict_time_steps=predict_time_steps,
            return_dict=True,
        )
        base_preds = base_out.preds
        base_step = base_preds[:, 0] if base_preds.dim() == 5 else base_preds
        labels_step = labels[:, 0] if (labels is not None and labels.dim() == 5) else labels

        if labels_step is None:
            if atmo_vars is None:
                raise ValueError("atmo_vars is required for perturbation inference.")
            cond = torch.cat([base_step, atmo_vars], dim=1)
            delta_hat = self.sample_delta(cond, base_step.shape)
            if base_preds.dim() == 5:
                preds = base_preds.clone()
                preds[:, 0] = preds[:, 0] + delta_hat
            else:
                preds = base_preds + delta_hat
            if not return_dict:
                return (preds,)
            return ORCADLOutput(loss=None, preds=preds)

        if atmo_vars is None:
            raise ValueError("atmo_vars is required for perturbation training.")

        delta = labels_step - base_step
        B = delta.shape[0]
        t = torch.randint(0, self.config.diffusion_steps, (B,), device=delta.device)
        noise = torch.randn_like(delta)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
        x_t = sqrt_alpha * delta + sqrt_one_minus * noise

        cond = torch.cat([base_step, atmo_vars], dim=1)
        noise_pred = self.noise_model(x_t, t, cond)
        loss = F.mse_loss(noise_pred, noise)

        if not return_dict:
            return (loss, base_preds)
        return ORCADLOutput(loss=loss, preds=base_preds)