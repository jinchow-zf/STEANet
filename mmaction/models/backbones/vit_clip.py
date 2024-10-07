import math
from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from mmaction.utils import get_root_logger
from einops import rearrange
from ..builder import BACKBONES


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

# Temporal Emotion Adapter
class TEA(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, self.D_hidden_features)
        self.conv1d = nn.Conv1d(in_channels=self.D_hidden_features, out_channels=self.D_hidden_features, kernel_size=1, stride=1,
                                padding=0)
        self.D_fc2 = nn.Linear(self.D_hidden_features, D_features)
        nn.init.constant_(self.conv1d.weight, 0.)
        nn.init.constant_(self.conv1d.bias, 0.)

    def forward(self, x, N: int = 197):
        # x is (T, BN, D)
        T, BN, D = x.size()
        B = BN // N
        C = self.D_hidden_features
        xs = self.D_fc1(x)
        xs = xs.view(B*N, C, T).contiguous()
        xs = self.conv1d(xs)
        xs = xs.view(T, B*N, C).contiguous()
        xs = self.D_fc2(xs)
        # xs = xs.view(N, B*T, D).contiguous()
        return xs


# Spatial Emotion Adapter
class SEA(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25):
        super().__init__()
        self.D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, self.D_hidden_features)
        self.D_fc2 = nn.Linear(self.D_hidden_features, D_features)
        self.conv2d = nn.Conv2d(
            self.D_hidden_features, self.D_hidden_features,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.D_hidden_features,
        )
        nn.init.constant_(self.conv2d.weight, 0.)
        nn.init.constant_(self.conv2d.bias, 0.)

    def forward(self, x):
        n, bt, d = x.size()
        Cm = self.D_hidden_features
        h = w = round(math.sqrt(n - 1))
        assert n - 1 == h * w
        x = x[1:, :, :]
        x = self.D_fc1(x)
        x = x.view(h, w, bt, Cm).permute(2, 3, 0, 1).contiguous()
        x = self.conv2d(x)
        x = x.permute(0, 1, 2, 3).contiguous().view(n - 1, bt, Cm)
        x = self.D_fc2(x)
        return x


# class STAdapter(nn.Module):
#     def __init__(self, in_channels=768, hidden_channels=384):
#         super().__init__()
#         self.fc1 = nn.Linear(in_channels, hidden_channels)
#         self.fc2 = nn.Linear(hidden_channels, in_channels)
#         self.conv = nn.Conv3d(
#             hidden_channels, hidden_channels,
#             kernel_size=(3, 1, 1),
#             stride=(1, 1, 1),
#             padding=tuple([1, 0, 0]),
#             groups=hidden_channels,
#         )
#         self.hidden_channels = hidden_channels
#         nn.init.constant_(self.conv.weight, 0.)
#         nn.init.constant_(self.conv.bias, 0.)
#
#     def forward(self, x, T: int = 32):
#         BT, L, C = x.size()
#         B = BT // T
#         Ca = self.hidden_channels
#         H = W = round(math.sqrt(L - 1))
#         assert L - 1 == H * W
#         x_id = x
#         x = x[:, 1:, :]
#
#         x = self.fc1(x)
#         x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()
#
#         x = self.conv(x)
#
#         x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
#         x = self.fc2(x)
#         x_id[:, 1:, :] += x
#         return x_id


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1,
                 num_frames=8, drop_path=0.):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        # self.ST_Adapter = STAdapter()
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        self.TEA = TEA(d_model)
        self.SEA = SEA(d_model)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape

        ## temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
        xt_ln = self.ln_1(xt)
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attention(self.T_Adapter_in(xt_ln)))
        else:
            xt = self.T_Adapter(self.attention(xt_ln))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x_conv1 = self.TEA(xt_ln).view(n, bt, d).contiguous()
        x = x + self.drop_path(xt) + x_conv1
        ## spatial adaptation
        xs_ln = self.ln_1(x)
        x_conv2 = self.SEA(xs_ln)
        xs = self.S_Adapter(self.attention(xs_ln))
        xs[1:, :, :] += x_conv2
        x = x + xs
        ## joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1,
                 scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i]) for i in
              range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@BACKBONES.register_module()
class ViT_CLIP(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int,
                 drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale,
                                       drop_path=drop_path_rate)

        self.ln_post = LayerNorm(width)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
        ## initialize TEA
        for n, m in self.transformer.named_modules():
            if 'TEA' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
        ## initialize SEA
        for n, m in self.transformer.named_modules():
            if 'SEA' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t', b=B, t=T)

        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x
