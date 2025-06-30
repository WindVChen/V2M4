import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from craftsman.utils.typing import *
from craftsman.utils.checkpoint import checkpoint

from .utils import init_linear, MLP
from timm.models.vision_transformer import Attention

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        qk_norm: bool = False,
        qkv_fuse: bool = True,
        norm_layer=nn.LayerNorm,
        use_flash: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.qkv_fuse = qkv_fuse
        if qkv_fuse:
            self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        else:
            self.c_q = nn.Linear(width, width, bias=qkv_bias)
            self.c_k = nn.Linear(width, width, bias=qkv_bias)
            self.c_v = nn.Linear(width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads, 
            n_ctx=n_ctx, 
            width=width, 
            norm_layer=norm_layer, 
            qk_norm=qk_norm, 
            use_flash=use_flash
        )
        if qkv_fuse:
            init_linear(self.c_qkv, init_scale)
        else:
            init_linear(self.c_q, init_scale)
            init_linear(self.c_k, init_scale)
            init_linear(self.c_v, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        if self.qkv_fuse:
            x = self.c_qkv(x)
            x = checkpoint(self.attention.forward_qkv_fuse, (x,), (), True)
        else:
            q = self.c_q(x)
            k = self.c_k(x)
            v = self.c_v(x)
            x = checkpoint(self.attention, (q,k,v,), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int, width=None, qk_norm: bool = False, norm_layer=nn.LayerNorm, use_flash: bool = False):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.use_flash = use_flash

        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, k, v):
        bs, n_ctx, width = q.shape
        attn_ch = width // self.heads
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        k = k.view(bs, n_ctx, self.heads, -1)
        v = v.view(bs, n_ctx, self.heads, -1)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out

    def forward_qkv_fuse(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        use_flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        qkv_fuse: bool = True, 
        use_flash: bool = False,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.qkv_fuse = qkv_fuse
        if qkv_fuse:
            self.c_q = nn.Linear(width, width, bias=qkv_bias)
            self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        else:
            self.c_q = nn.Linear(width, width, bias=qkv_bias)
            self.c_k = nn.Linear(width, width, bias=qkv_bias)
            self.c_v = nn.Linear(width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, width=width, norm_layer=norm_layer, qk_norm=qk_norm, use_flash=use_flash
        )
        if qkv_fuse:
            init_linear(self.c_q, init_scale)
            init_linear(self.c_kv, init_scale)
        else:
            init_linear(self.c_q, init_scale)
            init_linear(self.c_k, init_scale)
            init_linear(self.c_v, init_scale)

        init_linear(self.c_proj, init_scale)

    def forward(self, x, data):
        if self.qkv_fuse:
            x = self.c_q(x)
            data = self.c_kv(data)
            x = checkpoint(self.attention.forward_qkv_fuse, (x, data), (), True)
        else:
            q = self.c_q(x)
            k = self.c_k(data)
            v = self.c_v(data)
            x = checkpoint(self.attention, (q, k, v,), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self, 
        *, 
        heads: int, 
        n_data: Optional[int] = None, 
        width=None, 
        norm_layer=nn.LayerNorm, 
        qk_norm: bool = False, 
        use_flash: bool = False
    ):

        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.use_flash = use_flash
        
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, k, v):
        _, n_ctx, _ = q.shape
        bs, n_data, width = k.shape
        attn_ch = width // self.heads
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        k = k.view(bs, n_data, self.heads, -1)
        v = v.view(bs, n_data, self.heads, -1)

        if self.use_flash:
            
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out

    def forward_qkv_fuse(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_flash: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_flash=use_flash,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x