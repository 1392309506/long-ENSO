import logging
import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass
from timm.models.layers import trunc_normal_
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import ModelOutput, PreTrainedModel

from .layer import SwinLayer
logger = logging.getLogger(__name__)


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
        mask_patch_size=(8, 12),
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


class ORCADLPreTrainedModel(PreTrainedModel):
    config_class = ORCADLConfig
    main_input_name = "ocean_vars"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, SwinLayer):
            logger.info(
                "Set gradient_checkpointing=%s on %s",
                value,
                module.__class__.__name__,
            )
            module.gradient_checkpointing = value
