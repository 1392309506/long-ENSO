import torch
import math
import inspect
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .layer import DiffusionTimeEmbedding
from .base import ORCADLConfig, ORCADLOutput, BasePreTrainedModel
from .deepsea_model import BaseModel


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


class NoiseModel(nn.Module):
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


class PerturbationModel(BasePreTrainedModel):
    def __init__(
            self,
            config: ORCADLConfig,                     # 模型配置对象，包含各种超参数
            base_model: Optional[BaseModel] = None,   # Stage-1 的基础预测模型
            freeze_base_model: bool = True,           # 是否冻结 Stage-1 参数
    ):
        super().__init__(config)  # 调用父类初始化（通常用于保存 config 等）

        # -------------------------------------------------
        # BaseModel：Stage-1 backbone
        # -------------------------------------------------
        if base_model is None:
            self.backbone_model = BaseModel(config)
        else:
            self.backbone_model = base_model

        # -------------------------------------------------
        # 检查 BaseModel.forward 是否支持 atmo_vars
        # -------------------------------------------------

        # 读取 backbone_model 的 forward 方法
        base_forward = getattr(self.backbone_model, "forward", None)

        if base_forward is not None:
            # 获取 forward 函数的参数列表
            params = inspect.signature(base_forward).parameters

            # 判断 forward 是否包含 atmo_vars 参数
            self._base_accepts_atmo = "atmo_vars" in params
        else:
            # 如果没有 forward 方法，则默认不支持
            self._base_accepts_atmo = False

        # -------------------------------------------------
        # 冻结 Stage-1
        # -------------------------------------------------
        # Stage-2 训练时不更新 BaseModel
        # -------------------------------------------------

        if freeze_base_model:
            # 遍历 backbone_model 的所有参数
            for p in self.backbone_model.parameters():
                # 关闭梯度计算 → 不参与训练
                p.requires_grad = False

        # -------------------------------------------------
        # 变量配置
        # -------------------------------------------------
        self.stage2_full_vars = getattr(config, "stage2_full_vars", None)  # Stage-2 所有变量名称列表
        self.stage2_base_vars = getattr(config, "stage2_base_vars", None) # Stage-1 使用的变量（作为 base input）
        self.stage2_surface_vars = getattr(config, "stage2_surface_vars", None) # Stage-2 条件输入中的 surface 变量
        self.stage2_target_vars = getattr(config, "stage2_target_vars", None) # Stage-2 需要预测的变量
        self.stage2_var_chans = getattr(config, "stage2_var_chans", None) # 每个变量对应的 channel 数

        self._var_slices = None # 保存变量对应的 channel slice

        # -------------------------------------------------
        # 默认通道数
        # -------------------------------------------------

        self.base_chans = sum(config.out_chans) # BaseModel 输出通道数（所有变量 channel 之和）
        self.surface_chans = 0  # surface 变量默认 0
        self.target_chans = sum(config.out_chans) # diffusion 目标变量默认通道数

        # -------------------------------------------------
        # 构建变量 channel slice
        # -------------------------------------------------

        # 如果提供了完整变量列表和通道数
        if self.stage2_full_vars and self.stage2_var_chans:

            # 构建变量 → channel slice 映射
            # 例如 temp → slice(0,50)
            self._var_slices = self._build_var_slices(self.stage2_full_vars, self.stage2_var_chans)

            # 计算 base_vars 的总通道数
            if self.stage2_base_vars:
                self.base_chans = sum(self._slice_size(self._var_slices[v]) for v in self.stage2_base_vars)

            # 计算 surface_vars 的总通道数
            if self.stage2_surface_vars:
                self.surface_chans = sum(self._slice_size(self._var_slices[v]) for v in self.stage2_surface_vars)

            # 计算 target_vars 的总通道数
            if self.stage2_target_vars:
                self.target_chans = sum(self._slice_size(self._var_slices[v]) for v in self.stage2_target_vars)

        # -------------------------------------------------
        # diffusion 条件输入通道
        # -------------------------------------------------

        # diffusion 条件输入通道数：
        # base prediction + surface vars + atmosphere vars
        cond_chans = self.base_chans + self.surface_chans + config.atmo_dims

        # -------------------------------------------------
        # diffusion noise predictor
        # -------------------------------------------------

        # 扩散模型中的噪声预测网络 ε_θ(x_t, t, cond)
        self.noise_model = NoiseModel(
            in_chans=self.target_chans,                   # 输入 x_t 的通道数（目标变量）
            cond_chans=cond_chans,                        # 条件输入通道数
            base_chans=config.embed_dim,                  # 网络基础通道数
            time_embed_dim=config.diffusion_time_embed_dim,  # diffusion timestep embedding 维度
        )

        # -------------------------------------------------
        # diffusion schedule
        # -------------------------------------------------

        # 构造 beta schedule（噪声强度）
        betas = torch.linspace(
            config.diffusion_beta_start,   # beta 起始值
            config.diffusion_beta_end,     # beta 结束值
            config.diffusion_steps         # diffusion 总步数
        )

        # alpha_t = 1 - beta_t
        alphas = 1.0 - betas

        # ᾱ_t = ∏_{i=1..t} α_i
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 将这些张量注册为 buffer（不会训练，但会随模型保存）
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # √ᾱ_t
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))

        # √(1-ᾱ_t)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _p_mean_variance(self, x_t: torch.FloatTensor, t: torch.LongTensor, cond: torch.FloatTensor):

        # 预测噪声 ε_θ(x_t , t , cond)
        eps_pred = self.noise_model(x_t, t, cond)

        # 当前 timestep 的 beta_t
        beta_t = self.betas[t].view(-1, 1, 1, 1)

        # 当前 timestep 的 alpha_t
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)

        # √(1-ᾱ_t)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # 反向扩散均值公式（DDPM）
        # μ = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_pred)
        model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus) * eps_pred)

        return model_mean, beta_t

    def sample_delta(self, cond: torch.FloatTensor, shape: torch.Size):

        # 从标准高斯噪声开始
        x_t = torch.randn(shape, device=cond.device, dtype=cond.dtype)

        # 从 T → 0 逐步去噪
        for i in reversed(range(self.config.diffusion_steps)):

            # 构造 timestep tensor
            t = torch.full((shape[0],), i, device=cond.device, dtype=torch.long)

            # 计算 p(x_{t-1}|x_t)
            model_mean, beta_t = self._p_mean_variance(x_t, t, cond)

            if i > 0:
                # 添加随机噪声
                noise = torch.randn_like(x_t)

                # x_{t-1} = μ + √β_t * noise
                x_t = model_mean + torch.sqrt(beta_t) * noise
            else:
                # 最后一步不再加噪声
                x_t = model_mean

        return x_t

    def _build_var_slices(self, var_list, var_chans):

        # 构建变量 → channel slice 映射
        slices = {}

        start = 0

        for v, c in zip(var_list, var_chans):

            # 每个变量对应一个 channel slice
            slices[v] = slice(start, start + c)

            start += c

        return slices

    def _slice_size(self, slic):

        # 计算 slice 的大小
        return slic.stop - slic.start

    def _select_vars(self, x, var_names):

        # 如果没有 slice 信息则直接返回
        if self._var_slices is None:
            return x

        # 按变量名称提取 channel
        parts = [x[:, self._var_slices[v]] for v in var_names]

        # 在 channel 维拼接
        return torch.cat(parts, dim=1)

    def forward(
        self,
        ocean_vars: torch.FloatTensor,       # 海洋变量输入
        atmo_vars: torch.FloatTensor,        # 大气变量输入
        lead_time: Optional[torch.LongTensor] = None,
        atmo_lead_time: Optional[torch.LongTensor] = None,
        mask: torch.FloatTensor = None,
        labels: Optional[torch.FloatTensor] = None,   # 真实标签（训练时使用）
        predict_time_steps: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ):

        # 是否使用 dict 输出
        if return_dict is None:
            return_dict = self.config.use_return_dict

        # diffusion 模型参数 dtype
        model_dtype = self.noise_model.in_proj.weight.dtype

        # 如果指定了 base_vars 但没有 slice 信息
        if self.stage2_base_vars and self._var_slices is None:
            raise ValueError("stage2_full_vars and stage2_var_chans are required for Stage-2 conditioning.")

        # 提取 base 输入变量
        base_inputs = self._select_vars(ocean_vars, self.stage2_base_vars) if self.stage2_base_vars else ocean_vars

        surface_inputs = None

        # 提取 surface 变量
        if self.stage2_surface_vars:
            surface_inputs = self._select_vars(ocean_vars, self.stage2_surface_vars)

        print("[PerturbationModel]lead_time:", lead_time[:5])
        # 构造 BaseModel forward 参数
        base_kwargs = dict(
            ocean_vars=base_inputs,
            lead_time=lead_time,
            mask=mask,
            labels=None,
            predict_time_steps=predict_time_steps,
            return_dict=True,
        )

        # 如果 BaseModel 支持 atmo_vars
        if self._base_accepts_atmo:
            base_kwargs["atmo_vars"] = atmo_vars
            base_kwargs["atmo_lead_time"] = atmo_lead_time

        # 运行 Stage-1
        base_out = self.backbone_model(**base_kwargs)

        # 获取预测结果
        base_preds = base_out.preds

        # 如果包含时间维则取第 0 步
        base_step = base_preds[:, 0] if base_preds.dim() == 5 else base_preds

        # labels 同样取第一步
        labels_step = labels[:, 0] if (labels is not None and labels.dim() == 5) else labels

        # ---------------- 推理模式 ----------------
        if labels_step is None:

            if atmo_vars is None:
                raise ValueError("atmo_vars is required for perturbation inference.")

            if surface_inputs is None:
                raise ValueError("surface_vars are required for Stage-2 conditioning.")

            # diffusion 条件输入
            cond = torch.cat([base_step, surface_inputs, atmo_vars], dim=1).to(model_dtype)

            # 从 diffusion 采样生成 target
            target_hat = self.sample_delta(cond, (base_step.shape[0], self.target_chans) + base_step.shape[2:])

            preds = target_hat

            if not return_dict:
                return (preds,)

            return ORCADLOutput(loss=None, preds=preds)

        # ---------------- 训练模式 ----------------

        if atmo_vars is None:
            raise ValueError("atmo_vars is required for perturbation training.")

        if surface_inputs is None:
            raise ValueError("surface_vars are required for Stage-2 conditioning.")

        # 选择 target 变量
        target = self._select_vars(labels_step, self.stage2_target_vars) if self.stage2_target_vars else labels_step

        target = target.to(model_dtype)

        B = target.shape[0]

        # 随机采样 diffusion timestep
        t = torch.randint(0, self.config.diffusion_steps, (B,), device=target.device)

        # 真实噪声
        noise = torch.randn_like(target)

        # √ᾱ_t
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)

        # √(1-ᾱ_t)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        # forward diffusion:
        # x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * noise
        x_t = sqrt_alpha * target + sqrt_one_minus * noise

        # diffusion 条件输入
        cond = torch.cat([base_step, surface_inputs, atmo_vars], dim=1).to(model_dtype)

        # 预测噪声 ε_pred
        noise_pred = self.noise_model(x_t, t, cond)

        # diffusion 损失：预测噪声 MSE
        loss = F.mse_loss(noise_pred, noise)

        if not return_dict:
            return (loss, target)

        return ORCADLOutput(loss=loss, preds=target)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):

        model = super().from_pretrained(*args, **kwargs)

        has_backbone = any(
            k.startswith("backbone_model.")
            for k in model.state_dict().keys()
        )

        if not has_backbone:
            raise RuntimeError(
                "[ PerturbationModel ] Checkpoint does not contain BaseModel weights."
            )

        return model