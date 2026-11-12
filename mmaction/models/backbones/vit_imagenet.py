import math
from collections import OrderedDict

import clip
import torch
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

from mmaction.utils import get_root_logger
from ..builder import BACKBONES


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU,
                 skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, hidden)
        self.act = act_layer()
        self.D_fc2 = nn.Linear(hidden, D_features)

    def forward(self, x):
        y = self.D_fc2(self.act(self.D_fc1(x)))
        return x + y if self.skip_connect else y


class TEA(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU,
                 skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.D_hidden_features = int(D_features * mlp_ratio)
        c = self.D_hidden_features
        self.D_fc1 = nn.Linear(D_features, c)
        self.conv1d = nn.Conv1d(c, c, kernel_size=1)
        self.D_fc2 = nn.Linear(c, D_features)
        nn.init.zeros_(self.conv1d.weight)
        nn.init.zeros_(self.conv1d.bias)
        nn.init.zeros_(self.D_fc2.bias)

    def forward(self, x, N=None):
        if N is not None and (N <= 0 or x.size(1) % N != 0):
            raise ValueError("The second dimension of TEA input must be divisible by N.")
        y = self.D_fc1(x)
        y = y.permute(1, 2, 0).contiguous()
        y = self.conv1d(y)
        y = y.permute(2, 0, 1).contiguous()
        return self.D_fc2(y)


class SEA(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25):
        super().__init__()
        self.D_hidden_features = int(D_features * mlp_ratio)
        c = self.D_hidden_features
        self.D_fc1 = nn.Linear(D_features, c)
        self.conv2d = nn.Conv2d(c, c, kernel_size=1, groups=c)
        self.D_fc2 = nn.Linear(c, D_features)
        nn.init.zeros_(self.conv2d.weight)
        nn.init.zeros_(self.conv2d.bias)
        nn.init.zeros_(self.D_fc2.bias)

    def forward(self, x):
        m, bt, _ = x.shape
        h = int(math.sqrt(m))
        if m == 0 or h * h != m:
            raise ValueError("SEA expects a square grid of patch tokens without CLS.")
        c = self.D_hidden_features
        y = self.D_fc1(x)
        y = y.reshape(h, h, bt, c).permute(2, 3, 0, 1).contiguous()
        y = self.conv2d(y)
        y = y.permute(2, 3, 0, 1).contiguous().reshape(m, bt, c)
        return self.D_fc2(y)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).to(x.dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None, scale=1.,
                 num_tadapter=1, num_frames=8, drop_path=0.):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.num_frames = num_frames
        self.n_head = n_head
        self.scale = scale
        self.attn_mask = attn_mask
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        self.TEA = TEA(d_model)
        self.SEA = SEA(d_model)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(device=x.device, dtype=x.dtype)
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor):
        n = x.size(0)
        xt = rearrange(x, "n (b t) d -> t (b n) d", t=self.num_frames)
        xt_ln = self.ln_1(xt)
        attn_input = self.T_Adapter_in(xt_ln) if self.num_tadapter == 2 else xt_ln
        xt = self.T_Adapter(self.attention(attn_input))
        xt = rearrange(xt, "t (b n) d -> n (b t) d", n=n)
        tea = rearrange(self.TEA(xt_ln), "t (b n) d -> n (b t) d", n=n)
        x = x + self.drop_path(xt) + tea

        xs_ln = self.ln_1(x)
        sea = self.SEA(xs_ln[1:])
        xs = self.S_Adapter(self.attention(xs_ln))
        xs = torch.cat((xs[:1], xs[1:] + sea), dim=0)
        x = x + xs

        xn = self.ln_2(x)
        return x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int,
                 attn_mask=None, num_tadapter=1, scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        rates = torch.linspace(0, drop_path, layers).tolist()
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter,
                                   num_frames, rates[i])
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@BACKBONES.register_module()
class ViT_CLIP(nn.Module):
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int,
                 width: int, layers: int, heads: int, drop_path_rate,
                 num_tadapter=1, adapter_scale=0.5, pretrained=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_frames = num_frames
        self.layers = layers
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size,
                               stride=patch_size, bias=False)
        scale = width ** -0.5
        n = (input_resolution // patch_size) ** 2 + 1
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(n, width))
        self.ln_pre = LayerNorm(width)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.transformer = Transformer(num_frames, width, layers, heads,
                                       num_tadapter=num_tadapter, scale=adapter_scale,
                                       drop_path=drop_path_rate)
        self.ln_post = LayerNorm(width)

    def init_weights(self, pretrained=None):
        def init_module(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

        if pretrained is not None:
            self.pretrained = pretrained
        if self.pretrained is not None and not isinstance(self.pretrained, str):
            raise TypeError("pretrained must be a str or None")
        self.apply(init_module)

        if isinstance(self.pretrained, str):
            variants = {12: "ViT-B/16", 24: "ViT-L/14"}
            if self.layers not in variants:
                raise ValueError("OpenAI CLIP loading supports 12 or 24 layers here.")
            name = variants[self.layers]
            logger = get_root_logger()
            logger.info("Loading OpenAI CLIP visual weights: %s", name)
            clip_model, _ = clip.load(name, device="cpu")
            state = clip_model.visual.state_dict()
            state.pop("proj", None)
            msg = self.load_state_dict(state, strict=False)
            logger.info("Missing keys: %s", msg.missing_keys)
            logger.info("Unexpected keys: %s", msg.unexpected_keys)

        for module in self.transformer.modules():
            if isinstance(module, Adapter):
                nn.init.zeros_(module.D_fc2.weight)
                nn.init.zeros_(module.D_fc2.bias)
            elif isinstance(module, TEA):
                nn.init.zeros_(module.conv1d.weight)
                nn.init.zeros_(module.conv1d.bias)
            elif isinstance(module, SEA):
                nn.init.zeros_(module.conv2d.weight)
                nn.init.zeros_(module.conv2d.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed", "temporal_embedding"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table", "temporal_position_bias_table"}

    def forward(self, x: torch.Tensor):
        b, _, t, h, w = x.shape
        if t != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {t}.")
        if (h, w) != (self.input_resolution, self.input_resolution):
            raise ValueError(f"Expected square inputs of size {self.input_resolution}, got {h} x {w}.")

        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = rearrange(self.conv1(x), "bt d h w -> bt (h w) d")
        cls = self.class_embedding.to(x.dtype)[None, None, :].expand(x.size(0), 1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        n = x.size(1)
        x = rearrange(x, "(b t) n d -> (b n) t d", t=t)
        x = x + self.temporal_embedding.to(x.dtype)
        x = rearrange(x, "(b n) t d -> (b t) n d", n=n)
        x = self.ln_pre(x).permute(1, 0, 2)
        x = self.transformer(x).permute(1, 0, 2)
        x = self.ln_post(x)[:, 0]
        x = rearrange(x, "(b t) d -> b d t", b=b, t=t)
        return x.unsqueeze(-1).unsqueeze(-1)
