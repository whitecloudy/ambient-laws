import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch_utils import persistence


@torch.jit.script
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate real tensor x with shift and scale.
    x: [B, N, H] or [B, H]
    shift, scale: [B, H]
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def pad_to_power_of_two(x: torch.Tensor):
    """
    Pad spatial dimensions T and C to the nearest power of two.
    Supports 4D [B, C, T, W] or 5D [B, A, T, C, 2].
    """
    if x.ndim == 5:
        orig_H, orig_W = x.shape[2], x.shape[3]
    else:
        orig_H, orig_W = x.shape[-2], x.shape[-1]

    target_H = 1 << (orig_H - 1).bit_length() if orig_H > 0 else 0
    target_W = 1 << (orig_W - 1).bit_length() if orig_W > 0 else 0
    pad_h = target_H - orig_H
    pad_w = target_W - orig_W

    if pad_h > 0 or pad_w > 0:
        if x.ndim == 5:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

    return x, pad_h, pad_w, orig_H, orig_W


def unpad_from_power_of_two(x: torch.Tensor, pad_h: int, pad_w: int, orig_H: int, orig_W: int) -> torch.Tensor:
    """
    Unpad spatial dimensions T and C back to original shape.
    """
    if pad_h > 0 or pad_w > 0:
        if x.ndim == 5:
            return x[:, :, :orig_H, :orig_W, :]
        return x[..., :orig_H, :orig_W]
    return x


class RealLinear(nn.Linear):
    """
    Linear layer that casts weight and bias to input dtype to support mixed precision (e.g. BF16).
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(input.dtype)
        b = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, w, b)


class RealGroupNorm(nn.Module):
    """
    Group Normalization for real-valued tensors with shape [B, C, T, W].
    When causal=True, normalizes independently per time-step (B*T, C, 1, W)
    to strictly prevent future information leakage across the T dimension.
    """
    def __init__(self, num_channels: int, num_groups: int = 32, min_channels_per_group: int = 4, eps: float = 1e-5, causal: bool = False):
        super().__init__()
        self.causal = causal
        num_groups = min(num_groups, max(1, num_channels // min_channels_per_group))
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gn.weight.to(x.dtype) if self.gn.weight is not None else None
        b = self.gn.bias.to(x.dtype) if self.gn.bias is not None else None
        if self.causal:
            B, C, T, W = x.shape
            x_t = x.permute(0, 2, 1, 3).reshape(B * T, C, 1, W)
            out = F.group_norm(x_t, self.gn.num_groups, w, b, self.gn.eps).reshape(B, T, C, W).permute(0, 2, 1, 3)
            return out
        return F.group_norm(x, self.gn.num_groups, w, b, self.gn.eps)


class RealConv2d(nn.Module):
    """
    Real 2D Convolution operating on [B, C_in, T, W].
    Supports RF_SongUNet-style resample_filter for anti-aliasing downsampling and upsampling,
    as well as causal convolution along the T dimension.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias: bool = True,
        causal: bool = False,
        down: bool = False,
        up: bool = False,
        resample_filter = [1, 1],
    ):
        super().__init__()
        assert not (down and up), "Cannot specify both down=True and up=True"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.down = down
        self.up = up

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = tuple(dilation)

        # Setup resample_filter matching RF_SongUNet
        if (down or up) and resample_filter is not None:
            f_h = torch.as_tensor(resample_filter[0] if isinstance(resample_filter[0], (list, tuple)) else resample_filter, dtype=torch.float32)
            f_w = torch.as_tensor(resample_filter[1] if len(resample_filter) > 1 and isinstance(resample_filter[1], (list, tuple)) else resample_filter, dtype=torch.float32)
            f = f_h.ger(f_w)
            f = f.unsqueeze(0).unsqueeze(1) / (f_h.sum() * f_w.sum())
            self.register_buffer('resample_filter', f)
        else:
            self.register_buffer('resample_filter', None)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=self.kernel_size,
            stride=self.stride, dilation=self.dilation, padding=0, bias=bias
        )
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, W]
        # 1. FIR Downsampling / Upsampling matching RF_SongUNet
        if self.down and self.resample_filter is not None:
            f = self.resample_filter.to(x.dtype)
            kf_t, kf_c = f.shape[-2], f.shape[-1]
            if self.causal:
                pad_top_t = kf_t - 1
                pad_bottom_t = 0
                pad_left_c = 0
                pad_right_c = kf_c - 1
            else:
                pad_top_t = (kf_t - 1) // 2
                pad_bottom_t = kf_t - 1 - pad_top_t
                pad_left_c = (kf_c - 1) // 2
                pad_right_c = kf_c - 1 - pad_left_c

            x = F.pad(x, (pad_left_c, pad_right_c, pad_top_t, pad_bottom_t))
            f_tile = f.tile([self.in_channels, 1, 1, 1])
            x = F.conv2d(x, f_tile, groups=self.in_channels, stride=(2, 2))

        elif self.up and self.resample_filter is not None:
            f = self.resample_filter.to(x.dtype)
            f_tile = f.mul(4.0).tile([self.in_channels, 1, 1, 1])
            x = F.conv_transpose2d(x, f_tile, groups=self.in_channels, stride=(2, 2))

        # 2. Main convolution weights (w)
        k_t, k_c = self.kernel_size
        d_t, d_c = self.dilation

        pad_c_total = d_c * (k_c - 1)
        pad_c_left = pad_c_total // 2
        pad_c_right = pad_c_total - pad_c_left

        pad_t_total = d_t * (k_t - 1)
        if self.causal:
            pad_t_top = pad_t_total
            pad_t_bottom = 0
        else:
            pad_t_top = pad_t_total // 2
            pad_t_bottom = pad_t_total - pad_t_top

        if pad_c_total > 0 or pad_t_total > 0:
            x = F.pad(x, (pad_c_left, pad_c_right, pad_t_top, pad_t_bottom))

        w = self.conv.weight.to(x.dtype)
        b = self.conv.bias.to(x.dtype) if self.conv.bias is not None else None

        return F.conv2d(x, w, b, padding=0, stride=self.stride, dilation=self.dilation)


class RealResample2d(nn.Module):
    """
    Real Resampling layer for 2x downsampling or 2x upsampling with RF_SongUNet FIR filter.
    Includes residual projection matching RF_SongUNet.
    """
    def __init__(self, channels: int, down: bool = False, up: bool = False, causal: bool = False, resample_filter=[1, 1]):
        super().__init__()
        assert down ^ up, "Must specify exactly one of down=True or up=True"
        self.down = down
        self.up = up
        self.causal = causal
        self.conv = RealConv2d(
            channels, channels, kernel_size=3,
            down=down, up=up, resample_filter=resample_filter, causal=causal
        )
        self.skip = RealConv2d(
            channels, channels, kernel_size=1,
            down=down, up=up, resample_filter=resample_filter, causal=causal
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.conv(x) + self.skip(x)) * (1.0 / math.sqrt(2.0))


class RealUNetBlock(nn.Module):
    """
    Real ResNet Block with diffusion conditioning modulation.
    Supports causal 2D convolution on T.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        causal_conv: bool = False,
        dropout: float = 0.0,
        adaptive_scale: bool = True,
        kernel: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adaptive_scale = adaptive_scale

        self.norm0 = RealGroupNorm(in_channels, causal=causal_conv)
        self.act0 = nn.SiLU()
        self.conv0 = RealConv2d(in_channels, out_channels, kernel_size=kernel, causal=causal_conv)

        num_affine = 2 if adaptive_scale else 1
        self.affine = RealLinear(emb_channels, out_channels * num_affine)
        nn.init.constant_(self.affine.weight, 0)
        nn.init.constant_(self.affine.bias, 0)

        self.norm1 = RealGroupNorm(out_channels, causal=causal_conv)
        self.act1 = nn.SiLU()
        self.conv1 = RealConv2d(out_channels, out_channels, kernel_size=kernel, causal=causal_conv)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.skip = RealConv2d(in_channels, out_channels, kernel_size=1, causal=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, T, W]
        # emb: [B, emb_channels]
        orig = x

        h = self.conv0(self.act0(self.norm0(x)))

        params = self.affine(emb).to(x.dtype).view(x.shape[0], -1, 1, 1)
        h_norm = self.norm1(h)
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=1)
            h = self.act1(h_norm * (1.0 + scale) + shift)
        else:
            shift = params
            h = self.act1(h_norm + shift)

        h = self.conv1(self.dropout(h))
        return h + self.skip(orig)


class RealPositionalEmbedding2D(nn.Module):
    """
    Configurable real-valued positional embeddings for 2D token grid (T_len x W_len):
    1. 'learnable_2d': 2D Learnable real parameters.
    2. 'sincos_2d': 2D Sinusoidal Sine-Cosine positional embedding.
    3. 'hybrid_1d_t_sincos_c_learnable': 1D Sinusoidal for T, 1D Learnable for W (subcarrier/antenna).
    """
    def __init__(self, pos_embed_type: str, hidden_dim: int, max_t: int = 512, max_c: int = 256):
        super().__init__()
        self.pos_embed_type = pos_embed_type
        self.hidden_dim = hidden_dim
        self.max_t = max_t
        self.max_c = max_c

        if pos_embed_type == 'learnable_2d':
            self.pos_embed = nn.Parameter(torch.randn(1, max_t, max_c, hidden_dim) * 0.02)
        elif pos_embed_type == 'sincos_2d':
            pass
        elif pos_embed_type == 'hybrid_1d_t_sincos_c_learnable':
            self.c_pos_embed = nn.Parameter(torch.randn(1, 1, max_c, hidden_dim) * 0.02)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

    def forward(self, x: torch.Tensor, t_len: int, c_len: int) -> torch.Tensor:
        # x: [B, N, hidden_dim], where N = t_len * c_len
        B, N, H = x.shape
        device = x.device
        dtype = x.dtype

        if self.pos_embed_type == 'learnable_2d':
            pe = self.pos_embed[:, :t_len, :c_len, :]
            pe = pe.reshape(1, N, H).to(device=device, dtype=dtype)
            return x + pe

        elif self.pos_embed_type == 'sincos_2d':
            half_dim = H // 2
            rem_dim = H - half_dim

            # For T dimension (half_dim)
            half_t = half_dim // 2
            rem_t = half_dim - half_t
            t_steps = torch.arange(t_len, device=device, dtype=torch.float32).unsqueeze(1)
            t_dims = torch.arange(half_t, device=device, dtype=torch.float32).unsqueeze(0)
            t_freq = t_steps * torch.exp(-math.log(10000.0) * t_dims / max(1, half_t))
            t_sin = t_freq.sin()
            t_cos = t_freq.cos()
            if rem_t > half_t:
                t_emb = torch.cat([t_sin, t_cos, torch.zeros(t_len, 1, device=device)], dim=-1)
            else:
                t_emb = torch.cat([t_sin, t_cos], dim=-1)

            # For C dimension (rem_dim)
            half_c = rem_dim // 2
            rem_c = rem_dim - half_c
            c_steps = torch.arange(c_len, device=device, dtype=torch.float32).unsqueeze(1)
            c_dims = torch.arange(half_c, device=device, dtype=torch.float32).unsqueeze(0)
            c_freq = c_steps * torch.exp(-math.log(10000.0) * c_dims / max(1, half_c))
            c_sin = c_freq.sin()
            c_cos = c_freq.cos()
            if rem_c > half_c:
                c_emb = torch.cat([c_sin, c_cos, torch.zeros(c_len, 1, device=device)], dim=-1)
            else:
                c_emb = torch.cat([c_sin, c_cos], dim=-1)

            t_emb_2d = t_emb.unsqueeze(1).expand(t_len, c_len, half_dim)
            c_emb_2d = c_emb.unsqueeze(0).expand(t_len, c_len, rem_dim)
            pe = torch.cat([t_emb_2d, c_emb_2d], dim=-1).reshape(1, N, H).to(device=device, dtype=dtype)
            return x + pe

        elif self.pos_embed_type == 'hybrid_1d_t_sincos_c_learnable':
            half_h = H // 2
            rem_h = H - half_h
            t_steps = torch.arange(t_len, device=device, dtype=torch.float32).unsqueeze(1)
            dims = torch.arange(half_h, device=device, dtype=torch.float32).unsqueeze(0)
            t_freq = t_steps * torch.exp(-math.log(10000.0) * dims / max(1, half_h))
            t_sin = t_freq.sin()
            t_cos = t_freq.cos()
            if rem_h > half_h:
                t_emb = torch.cat([t_sin, t_cos, torch.zeros(t_len, 1, device=device)], dim=-1)
            else:
                t_emb = torch.cat([t_sin, t_cos], dim=-1)
            t_pe = t_emb.unsqueeze(1).expand(t_len, c_len, H).reshape(1, N, H).to(device=device, dtype=dtype)

            c_pe = self.c_pos_embed[:, :, :c_len, :].expand(1, t_len, c_len, H).reshape(1, N, H).to(device=device, dtype=dtype)
            return x + t_pe + c_pe

        return x


class RealMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using PyTorch scaled_dot_product_attention (SDPA).
    Enables FlashAttention / memory-efficient attention automatically on CUDA.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.qkv_proj = RealLinear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.out_proj = RealLinear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, H]
        B, N, H = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        out = out.permute(0, 2, 1, 3).reshape(B, N, H)
        return self.out_proj(out)


class RealDiTBlock(nn.Module):
    """
    Real DiT Transformer block with adaLN-Zero modulation.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = RealMultiHeadAttention(hidden_dim, num_heads, dropout=dropout, bias=True)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            RealLinear(hidden_dim, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            RealLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            RealLinear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, N, hidden_dim]
        # c: [B, hidden_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(mod_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mod_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(mod_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class RealFinalLayer(nn.Module):
    """
    Final projection layer of DiT with adaLN-Zero modulation.
    """
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = RealLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            RealLinear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1.0 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, scale: float = 16.0):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.ger((2.0 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


@persistence.persistent_class
class RF_DiT_Real(nn.Module, PyTorchModelHubMixin):
    """
    RF_DiT_Real: Pure real-valued Diffusion Transformer with Convolutional Encoder-Decoder
    and U-ViT style skip connections for RF / CSI data.
    Takes input of shape [B, 2*A, T, C] where real and imaginary components are concatenated
    along the channel/antenna dimension. Also accepts legacy 5D [B, A, T, C, 2] tensors seamlessly.
    """
    def __init__(
        self,
        img_resolution,
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        label_resolution = None,
        label_type: str = 'classes',
        augment_dim: int = 0,
        model_channels: int = 64,
        encdec_channel_mult = [1, 2],
        channel_mult = None,
        channel_mult_emb: int = 4,
        channel_mult_noise: int = 1,
        dropout: float = 0.1,
        class_dropout_rate: float = 0.1,
        num_blocks_per_stage: int = 4,
        causal_conv: bool = True,
        num_dit_blocks: int = 20,
        dit_hidden_dim: int = 128,
        dit_num_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        pos_embed_type: str = 'learnable_2d',
        embedding_type: str = 'positional',
        encoder_type: str = 'standard',
        decoder_type: str = 'standard',
        resample_filter = [1, 1],
        **kwargs
    ):
        super().__init__()

        if channel_mult is not None:
            encdec_channel_mult = channel_mult

        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.label_type = label_type
        self.label_dropout = class_dropout_rate
        self.causal_conv = causal_conv
        self.encdec_channel_mult = encdec_channel_mult
        self.dit_hidden_dim = dit_hidden_dim

        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise

        # Noise timestep embedding
        if embedding_type == 'positional':
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        else:
            self.map_noise = FourierEmbedding(num_channels=noise_channels)

        # Class / condition label embedding
        if label_dim != 0 and label_type == 'classes':
            self.map_label = RealLinear(label_dim, noise_channels)
        elif label_dim != 0 and label_type == 'downlink':
            self.map_label = nn.Sequential(
                RealLinear(label_dim, noise_channels * 2),
                nn.SiLU(),
                RealLinear(noise_channels * 2, noise_channels * 2),
            )
        else:
            self.map_label = None

        self.map_augment = RealLinear(augment_dim, noise_channels, bias=False) if augment_dim else None

        in_emb = noise_channels * (3 if (self.map_label is not None and label_type == 'downlink') else 1)
        self.map_layer0 = RealLinear(in_emb, emb_channels * 2)
        self.map_layer1 = RealLinear(emb_channels * 2, emb_channels)

        # Stem convolution
        self.stem = RealConv2d(in_channels, model_channels, kernel_size=3, causal=causal_conv)

        # Encoder stages
        self.enc_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        c_curr = model_channels
        for level, mult in enumerate(encdec_channel_mult):
            c_target = model_channels * mult
            stage_blocks = nn.ModuleList()
            for b_idx in range(num_blocks_per_stage):
                cin = c_curr if b_idx == 0 else c_target
                stage_blocks.append(
                    RealUNetBlock(
                        cin, c_target, emb_channels,
                        causal_conv=causal_conv, dropout=dropout
                    )
                )
            self.enc_blocks.append(stage_blocks)
            c_curr = c_target

            if level < len(encdec_channel_mult) - 1:
                self.downsamplers.append(RealResample2d(c_curr, down=True, causal=causal_conv, resample_filter=resample_filter))

        c_bottleneck = c_curr

        # Tokenization & DiT projection
        self.token_proj = RealLinear(c_bottleneck, dit_hidden_dim)
        H_res = img_resolution[0] if isinstance(img_resolution, (list, tuple)) else img_resolution
        W_res = img_resolution[1] if isinstance(img_resolution, (list, tuple)) else img_resolution
        max_t = 1 << int(np.ceil(np.log2(max(1, H_res))))
        max_c = 1 << int(np.ceil(np.log2(max(1, W_res))))
        self.pos_embed = RealPositionalEmbedding2D(pos_embed_type, dit_hidden_dim, max_t=max_t, max_c=max_c)

        # Projection from real emb to conditioning for DiT adaLN-Zero
        self.c_proj = RealLinear(emb_channels, dit_hidden_dim)

        # DiT Blocks
        self.dit_blocks = nn.ModuleList([
            RealDiTBlock(dit_hidden_dim, dit_num_heads, dropout=dropout, mlp_ratio=dit_mlp_ratio)
            for _ in range(num_dit_blocks)
        ])
        self.final_layer = RealFinalLayer(dit_hidden_dim, c_bottleneck)

        # Decoder stages
        self.dec_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        # Mirror levels in reverse
        for level in reversed(range(len(encdec_channel_mult))):
            mult = encdec_channel_mult[level]
            c_stage = model_channels * mult

            if level < len(encdec_channel_mult) - 1:
                self.upsamplers.append(RealResample2d(c_curr, up=True, causal=causal_conv, resample_filter=resample_filter))

            # Concatenation of skip: input channels = c_curr + c_stage
            cin_first = c_curr + c_stage
            stage_blocks = nn.ModuleList()
            for b_idx in range(num_blocks_per_stage):
                cin = cin_first if b_idx == 0 else c_stage
                stage_blocks.append(
                    RealUNetBlock(
                        cin, c_stage, emb_channels,
                        causal_conv=causal_conv, dropout=dropout
                    )
                )
            self.dec_blocks.append(stage_blocks)
            c_curr = c_stage

        # Final output projection
        self.out_conv = RealConv2d(c_curr, out_channels, kernel_size=3, causal=causal_conv)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor, class_labels=None, augment_labels=None, z=None, **kwargs):
        """
        Args:
            x: Real tensor [B, 2*A, T, C] or legacy 5D [B, A, T, C, 2]
            noise_labels: Diffusion noise level [B] or spatial noise [B, ...]
            class_labels: Class labels or condition tensor
        """
        is_5d_input = False
        orig_A = 0
        if x.ndim == 5 and x.shape[-1] == 2:
            is_5d_input = True
            orig_A = x.shape[1]
            x = torch.cat([x[..., 0], x[..., 1]], dim=1)

        # Pad spatial dimensions to power of two
        x, pad_h, pad_w, orig_H, orig_W = pad_to_power_of_two(x)

        # Process noise labels
        if noise_labels.ndim > 1:
            noise_labels = torch.sqrt(torch.mean(torch.flatten(noise_labels, start_dim=1)**2, dim=1))
        if noise_labels.ndim == 0:
            noise_labels = noise_labels.unsqueeze(0).repeat(x.shape[0])
        elif noise_labels.shape[0] == 1 and x.shape[0] > 1:
            noise_labels = noise_labels.repeat(x.shape[0])
        emb = self.map_noise(noise_labels.to(torch.float32))
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos

        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels.to(torch.float32))

        if self.map_label is not None:
            tmp = class_labels
            if (tmp is None or (isinstance(tmp, torch.Tensor) and tmp.ndim >= 2 and tmp.shape[-1] == 0)) and z is not None:
                tmp = z
            if tmp is not None:
                tmp = tmp.to(torch.float32)
                if self.training and self.label_dropout and self.label_type != 'no_label':
                    dropout_mask = (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                    while dropout_mask.ndim < tmp.ndim:
                        dropout_mask = dropout_mask.unsqueeze(-1)
                    tmp = tmp * dropout_mask
                if self.label_type == 'classes':
                    label_emb = self.map_label(tmp * np.sqrt(self.map_label.in_features))
                    emb = emb + label_emb
                elif self.label_type == 'downlink':
                    label_emb = self.map_label(tmp)
                    emb = torch.cat([emb, label_emb], dim=1)

        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))

        # 1. Stem
        x = self.stem(x)

        # 2. Convolutional Encoder with Skips
        skips = []
        down_idx = 0
        for level, stage in enumerate(self.enc_blocks):
            for block in stage:
                x = block(x, emb)
            skips.append(x)
            if level < len(self.enc_blocks) - 1:
                x = self.downsamplers[down_idx](x)
                down_idx += 1

        # 3. Patchification / Tokenization for DiT
        B, C_enc, t_down, c_down = x.shape
        N = t_down * c_down

        x_tokens = x.permute(0, 2, 3, 1).reshape(B, N, C_enc)
        x_tokens = self.token_proj(x_tokens)
        x_tokens = self.pos_embed(x_tokens, t_down, c_down)

        # DiT Conditioning
        c_dit = self.c_proj(emb).to(x.dtype)

        for dit_block in self.dit_blocks:
            x_tokens = dit_block(x_tokens, c_dit)

        x_tokens = self.final_layer(x_tokens, c_dit)

        # Unpatchify back to spatial grid
        x = x_tokens.reshape(B, t_down, c_down, C_enc).permute(0, 3, 1, 2)

        # 4. Convolutional Decoder with U-ViT Skips
        up_idx = 0
        for level_idx, stage in enumerate(self.dec_blocks):
            if level_idx > 0:
                x = self.upsamplers[up_idx](x)
                up_idx += 1
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            for block in stage:
                x = block(x, emb)

        # 5. Output Conv and unpad
        out = self.out_conv(x)
        out = unpad_from_power_of_two(out, pad_h, pad_w, orig_H, orig_W)

        if is_5d_input:
            out = torch.stack([out[:, :orig_A], out[:, orig_A:]], dim=-1)

        return out
