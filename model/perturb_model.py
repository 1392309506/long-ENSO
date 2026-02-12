import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .layer import DiffusionTimeEmbedding
from .base import ORCADLConfig, ORCADLOutput, ORCADLPreTrainedModel
from .deepsea_model import ORCADLModel


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
        h = torch.cat([x_t, cond], dim=1)
        h = self.in_proj(h)
        t_emb = self.time_embed(_timestep_embedding(t, self.time_embed.mlp[0].in_features))
        h = h + t_emb[:, :, None, None]
        h = self.res1(h)
        h = self.res2(h)
        return self.out_proj(h)


class ORCADLPerturbationModel(ORCADLPreTrainedModel):
    def __init__(
        self,
        config: ORCADLConfig,
        base_model: Optional[ORCADLModel] = None,
        freeze_base_model: bool = True,
    ):
        super().__init__(config)
        self.base_model = base_model if base_model is not None else ORCADLModel(config)
        if freeze_base_model:
            for p in self.base_model.parameters():
                p.requires_grad = False

        self.stage2_full_vars = getattr(config, "stage2_full_vars", None)
        self.stage2_base_vars = getattr(config, "stage2_base_vars", None)
        self.stage2_surface_vars = getattr(config, "stage2_surface_vars", None)
        self.stage2_target_vars = getattr(config, "stage2_target_vars", None)
        self.stage2_var_chans = getattr(config, "stage2_var_chans", None)

        self._var_slices = None
        self.base_chans = sum(config.out_chans)
        self.surface_chans = 0
        self.target_chans = sum(config.out_chans)

        if self.stage2_full_vars and self.stage2_var_chans:
            self._var_slices = self._build_var_slices(self.stage2_full_vars, self.stage2_var_chans)
            if self.stage2_base_vars:
                self.base_chans = sum(self._slice_size(self._var_slices[v]) for v in self.stage2_base_vars)
            if self.stage2_surface_vars:
                self.surface_chans = sum(self._slice_size(self._var_slices[v]) for v in self.stage2_surface_vars)
            if self.stage2_target_vars:
                self.target_chans = sum(self._slice_size(self._var_slices[v]) for v in self.stage2_target_vars)

        cond_chans = self.base_chans + self.surface_chans + config.atmo_dims
        self.noise_model = PerturbationModel(
            in_chans=self.target_chans,
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

    def _build_var_slices(self, var_list, var_chans):
        slices = {}
        start = 0
        for v, c in zip(var_list, var_chans):
            slices[v] = slice(start, start + c)
            start += c
        return slices

    def _slice_size(self, slic):
        return slic.stop - slic.start

    def _select_vars(self, x, var_names):
        if self._var_slices is None:
            return x
        parts = [x[:, self._var_slices[v]] for v in var_names]
        return torch.cat(parts, dim=1)

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

        if self.stage2_base_vars and self._var_slices is None:
            raise ValueError("stage2_full_vars and stage2_var_chans are required for Stage-2 conditioning.")

        base_inputs = self._select_vars(ocean_vars, self.stage2_base_vars) if self.stage2_base_vars else ocean_vars
        surface_inputs = None
        if self.stage2_surface_vars:
            surface_inputs = self._select_vars(ocean_vars, self.stage2_surface_vars)

        base_out = self.base_model(
            ocean_vars=base_inputs,
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
            if surface_inputs is None:
                raise ValueError("surface_vars are required for Stage-2 conditioning.")
            cond = torch.cat([base_step, surface_inputs, atmo_vars], dim=1)
            target_hat = self.sample_delta(cond, (base_step.shape[0], self.target_chans) + base_step.shape[2:])
            preds = target_hat
            if not return_dict:
                return (preds,)
            return ORCADLOutput(loss=None, preds=preds)

        if atmo_vars is None:
            raise ValueError("atmo_vars is required for perturbation training.")

        if surface_inputs is None:
            raise ValueError("surface_vars are required for Stage-2 conditioning.")
        target = self._select_vars(labels_step, self.stage2_target_vars) if self.stage2_target_vars else labels_step
        B = target.shape[0]
        t = torch.randint(0, self.config.diffusion_steps, (B,), device=target.device)
        noise = torch.randn_like(target)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
        x_t = sqrt_alpha * target + sqrt_one_minus * noise

        cond = torch.cat([base_step, surface_inputs, atmo_vars], dim=1)
        noise_pred = self.noise_model(x_t, t, cond)
        loss = F.mse_loss(noise_pred, noise)

        if not return_dict:
            return (loss, target)
        return ORCADLOutput(loss=loss, preds=target)
