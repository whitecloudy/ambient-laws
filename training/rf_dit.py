import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch_utils import persistence

from .complex import complex_module as cm


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate complex tensor x with complex shift and scale.
    x: [B, N, H, 2] or [B, H, 2]
    shift, scale: [B, H, 2]
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def pad_to_power_of_two(x: torch.Tensor):
    """
    Pad spatial dimensions T and C to the nearest power of two.
    Supports 5D [B, A, T, C, 2] or 4D [B, A, T, C].
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


class NaiveComplexGroupNorm(nn.Module):
    """
    Group Normalization for complex-valued tensors with shape [B, C, T, C_sub, 2].
    Applies GroupNorm independently to real and imaginary channels.
    When causal=True, normalizes independently per time-step (B*T, C, 1, C_sub)
    to strictly prevent future information leakage across the T dimension.
    """
    def __init__(self, num_channels: int, num_groups: int = 32, min_channels_per_group: int = 4, eps: float = 1e-5, causal: bool = False):
        super().__init__()
        self.causal = causal
        num_groups = min(num_groups, max(1, num_channels // min_channels_per_group))
        self.gn_r = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.gn_i = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x.unbind(-1)
        w_r = self.gn_r.weight.to(x_r.dtype) if self.gn_r.weight is not None else None
        b_r = self.gn_r.bias.to(x_r.dtype) if self.gn_r.bias is not None else None
        w_i = self.gn_i.weight.to(x_i.dtype) if self.gn_i.weight is not None else None
        b_i = self.gn_i.bias.to(x_i.dtype) if self.gn_i.bias is not None else None
        if self.causal:
            B, C, T, W = x_r.shape
            x_r_t = x_r.permute(0, 2, 1, 3).reshape(B * T, C, 1, W)
            x_i_t = x_i.permute(0, 2, 1, 3).reshape(B * T, C, 1, W)
            out_r = F.group_norm(x_r_t, self.gn_r.num_groups, w_r, b_r, self.gn_r.eps).reshape(B, T, C, W).permute(0, 2, 1, 3)
            out_i = F.group_norm(x_i_t, self.gn_i.num_groups, w_i, b_i, self.gn_i.eps).reshape(B, T, C, W).permute(0, 2, 1, 3)
        else:
            out_r = F.group_norm(x_r, self.gn_r.num_groups, w_r, b_r, self.gn_r.eps)
            out_i = F.group_norm(x_i, self.gn_i.num_groups, w_i, b_i, self.gn_i.eps)
        return torch.stack([out_r, out_i], dim=-1)


class ComplexConv2d(nn.Module):
    """
    Complex 2D Convolution operating on [B, C_in, T, C_sub, 2].
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
        resample_stride = 2,
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

        if isinstance(resample_stride, int):
            self.resample_stride = (resample_stride, resample_stride)
        else:
            self.resample_stride = tuple(resample_stride)

        # Setup resample_filter matching RF_SongUNet
        if (down or up) and resample_filter is not None:
            if isinstance(resample_filter, (list, tuple)) and len(resample_filter) == 2 and isinstance(resample_filter[0], (list, tuple)):
                f_h = torch.as_tensor(resample_filter[0], dtype=torch.float32)
                f_w = torch.as_tensor(resample_filter[1], dtype=torch.float32)
            else:
                f_h = torch.as_tensor(resample_filter, dtype=torch.float32)
                f_w = f_h
            if self.resample_stride[0] == 1:
                f_h = torch.ones(1, dtype=torch.float32)
            if self.resample_stride[1] == 1:
                f_w = torch.ones(1, dtype=torch.float32)
            f = f_h.ger(f_w)
            f = f.unsqueeze(0).unsqueeze(1) / (f_h.sum() * f_w.sum())
            self.register_buffer('resample_filter', f)
        else:
            self.register_buffer('resample_filter', None)

        self.conv_r = nn.Conv2d(
            in_channels, out_channels, kernel_size=self.kernel_size,
            stride=self.stride, dilation=self.dilation, padding=0, bias=bias
        )
        self.conv_i = nn.Conv2d(
            in_channels, out_channels, kernel_size=self.kernel_size,
            stride=self.stride, dilation=self.dilation, padding=0, bias=bias
        )

        nn.init.kaiming_normal_(self.conv_r.weight)
        nn.init.kaiming_normal_(self.conv_i.weight)
        if bias:
            nn.init.constant_(self.conv_r.bias, 0)
            nn.init.constant_(self.conv_i.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, C_sub, 2]
        x_r, x_i = x.unbind(-1)

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

            if pad_left_c > 0 or pad_right_c > 0 or pad_top_t > 0 or pad_bottom_t > 0:
                x_r = F.pad(x_r, (pad_left_c, pad_right_c, pad_top_t, pad_bottom_t))
                x_i = F.pad(x_i, (pad_left_c, pad_right_c, pad_top_t, pad_bottom_t))
            f_tile = f.tile([self.in_channels, 1, 1, 1])
            x_r = F.conv2d(x_r, f_tile, groups=self.in_channels, stride=self.resample_stride)
            x_i = F.conv2d(x_i, f_tile, groups=self.in_channels, stride=self.resample_stride)

        elif self.up and self.resample_filter is not None:
            f = self.resample_filter.to(x.dtype)
            stride_mul = float(self.resample_stride[0] * self.resample_stride[1])
            f_tile = f.mul(stride_mul).tile([self.in_channels, 1, 1, 1])
            x_r = F.conv_transpose2d(x_r, f_tile, groups=self.in_channels, stride=self.resample_stride)
            x_i = F.conv_transpose2d(x_i, f_tile, groups=self.in_channels, stride=self.resample_stride)

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
            x_r = F.pad(x_r, (pad_c_left, pad_c_right, pad_t_top, pad_t_bottom))
            x_i = F.pad(x_i, (pad_c_left, pad_c_right, pad_t_top, pad_t_bottom))

        w_r = self.conv_r.weight.to(x_r.dtype)
        b_r = self.conv_r.bias.to(x_r.dtype) if self.conv_r.bias is not None else None
        w_i = self.conv_i.weight.to(x_i.dtype)
        b_i = self.conv_i.bias.to(x_i.dtype) if self.conv_i.bias is not None else None

        out_r = F.conv2d(x_r, w_r, b_r, padding=0, stride=self.stride, dilation=self.dilation) - \
                F.conv2d(x_i, w_i, b_i, padding=0, stride=self.stride, dilation=self.dilation)
        out_i = F.conv2d(x_i, w_r, b_r, padding=0, stride=self.stride, dilation=self.dilation) + \
                F.conv2d(x_r, w_i, b_i, padding=0, stride=self.stride, dilation=self.dilation)

        return torch.stack([out_r, out_i], dim=-1)



class ComplexResample2d(nn.Module):
    """
    Complex Resampling layer for 2x downsampling or 2x upsampling with RF_SongUNet FIR filter.
    Includes residual projection matching RF_SongUNet.
    """
    def __init__(self, channels: int, down: bool = False, up: bool = False, causal: bool = False, resample_filter=[1, 1], resample_stride=(2, 2)):
        super().__init__()
        assert down ^ up, "Must specify exactly one of down=True or up=True"
        self.down = down
        self.up = up
        self.causal = causal
        self.resample_stride = resample_stride
        self.conv = ComplexConv2d(
            channels, channels, kernel_size=3,
            down=down, up=up, resample_filter=resample_filter, causal=causal,
            resample_stride=resample_stride
        )
        self.skip = ComplexConv2d(
            channels, channels, kernel_size=1,
            down=down, up=up, resample_filter=resample_filter, causal=causal,
            resample_stride=resample_stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.conv(x) + self.skip(x)) * (1.0 / math.sqrt(2.0))


class ComplexUNetBlock(nn.Module):
    """
    Complex ResNet Block with diffusion conditioning modulation.
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

        self.norm0 = NaiveComplexGroupNorm(in_channels, causal=causal_conv)
        self.act0 = cm.ComplexSiLU()
        self.conv0 = ComplexConv2d(in_channels, out_channels, kernel_size=kernel, causal=causal_conv)

        num_affine = 4 if adaptive_scale else 2
        self.affine = nn.Linear(emb_channels, out_channels * num_affine)
        nn.init.constant_(self.affine.weight, 0)
        nn.init.constant_(self.affine.bias, 0)

        self.norm1 = NaiveComplexGroupNorm(out_channels, causal=causal_conv)
        self.act1 = cm.ComplexSiLU()
        self.conv1 = ComplexConv2d(out_channels, out_channels, kernel_size=kernel, causal=causal_conv)
        self.dropout = cm.ComplexDropout(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.skip = ComplexConv2d(in_channels, out_channels, kernel_size=1, causal=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, T, C_sub, 2]
        # emb: [B, emb_channels]
        orig = x

        h = self.conv0(self.act0(self.norm0(x)))

        params = self.affine(emb).to(x.dtype).view(x.shape[0], self.out_channels, 1, 1, -1)
        h_norm = self.norm1(h)
        if self.adaptive_scale:
            scale = params[..., :2]
            shift = params[..., 2:]
            h = self.act1(h_norm * (1.0 + scale) + shift)
        else:
            shift = params[..., :2]
            h = self.act1(h_norm + shift)

        h = self.conv1(self.dropout(h))
        return h + self.skip(orig)


class ComplexPositionalEmbedding2D(nn.Module):
    """
    Configurable complex positional embeddings for 2D token grid (T_len x C_len):
    1. 'learnable_2d': 2D Learnable complex parameters.
    2. 'sincos_2d': 2D Complex Sine-Cosine phase rotation.
    3. 'hybrid_1d_t_sincos_c_learnable': 1D Sine-Cosine phase for T, 1D Learnable for C.
    """
    def __init__(self, pos_embed_type: str, hidden_dim: int, max_t: int = 512, max_c: int = 256):
        super().__init__()
        self.pos_embed_type = pos_embed_type
        self.hidden_dim = hidden_dim
        self.max_t = max_t
        self.max_c = max_c

        if pos_embed_type == 'learnable_2d':
            self.pos_embed = nn.Parameter(torch.randn(1, max_t, max_c, hidden_dim, 2) * 0.02)
        elif pos_embed_type == 'sincos_2d':
            pass
        elif pos_embed_type == 'hybrid_1d_t_sincos_c_learnable':
            self.c_pos_embed = nn.Parameter(torch.randn(1, 1, max_c, hidden_dim, 2) * 0.02)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

    def forward(self, x: torch.Tensor, t_len: int, c_len: int) -> torch.Tensor:
        # x: [B, N, hidden_dim, 2], where N = t_len * c_len
        B, N, H, _ = x.shape
        device = x.device
        dtype = x.dtype

        if self.pos_embed_type == 'learnable_2d':
            pe = self.pos_embed[:, :t_len, :c_len, :, :]
            pe = pe.reshape(1, N, H, 2).to(device=device, dtype=dtype)
            return x + pe

        elif self.pos_embed_type == 'sincos_2d':
            half_dim = H // 2
            rem_dim = H - half_dim

            t_steps = torch.arange(t_len, device=device, dtype=torch.float32).unsqueeze(1)
            t_dims = torch.arange(half_dim, device=device, dtype=torch.float32).unsqueeze(0)
            t_freq = t_steps * torch.exp(-math.log(10000.0) * t_dims / max(1, half_dim))

            c_steps = torch.arange(c_len, device=device, dtype=torch.float32).unsqueeze(1)
            c_dims = torch.arange(rem_dim, device=device, dtype=torch.float32).unsqueeze(0)
            c_freq = c_steps * torch.exp(-math.log(10000.0) * c_dims / max(1, rem_dim))

            t_freq_2d = t_freq.unsqueeze(1).expand(t_len, c_len, half_dim)
            c_freq_2d = c_freq.unsqueeze(0).expand(t_len, c_len, rem_dim)
            table = torch.cat([t_freq_2d, c_freq_2d], dim=-1).reshape(1, N, H)
            pe = torch.stack([table.cos(), table.sin()], dim=-1).to(device=device, dtype=dtype)
            return cm.complex_mul(x, pe)

        elif self.pos_embed_type == 'hybrid_1d_t_sincos_c_learnable':
            t_steps = torch.arange(t_len, device=device, dtype=torch.float32).unsqueeze(1)
            dims = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(0)
            t_freq = t_steps * torch.exp(-math.log(10000.0) * dims / H)
            t_pe = torch.stack([t_freq.cos(), t_freq.sin()], dim=-1).to(device=device, dtype=dtype)
            t_pe = t_pe.unsqueeze(1).expand(t_len, c_len, H, 2).reshape(1, N, H, 2)

            c_pe = self.c_pos_embed[:, :, :c_len, :, :].expand(1, t_len, c_len, H, 2).reshape(1, N, H, 2).to(device=device, dtype=dtype)
            x = x + c_pe
            return cm.complex_mul(x, t_pe)

        return x


class DiTBlock(nn.Module):
    """
    Complex DiT Transformer block with adaLN-Zero modulation.
    Based on wifi_model.py DiA block architecture.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = cm.ComplexMultiHeadAttention(hidden_dim, hidden_dim, num_heads, dropout=dropout, bias=True)
        self.norm2 = cm.NaiveComplexLayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[1].l_r.weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].l_i.weight, 0)
        if self.adaLN_modulation[1].l_r.bias is not None:
            nn.init.constant_(self.adaLN_modulation[1].l_r.bias, 0)
            nn.init.constant_(self.adaLN_modulation[1].l_i.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, N, hidden_dim, 2]
        # c: [B, hidden_dim, 2]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(mod_x, mod_x, mod_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mod_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(mod_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
    """
    Final projection layer of DiT with complex adaLN-Zero modulation.
    """
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[1].l_r.weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].l_i.weight, 0)
        if self.adaLN_modulation[1].l_r.bias is not None:
            nn.init.constant_(self.adaLN_modulation[1].l_r.bias, 0)
            nn.init.constant_(self.adaLN_modulation[1].l_i.bias, 0)
        nn.init.constant_(self.linear.l_r.weight, 0)
        nn.init.constant_(self.linear.l_i.weight, 0)
        if self.linear.l_r.bias is not None:
            nn.init.constant_(self.linear.l_r.bias, 0)
            nn.init.constant_(self.linear.l_i.bias, 0)

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
class RF_DiT(nn.Module, PyTorchModelHubMixin):
    """
    RF_DiT: Complex-valued Diffusion Transformer with Convolutional Encoder-Decoder
    and U-ViT style skip connections for RF / CSI data.
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
            self.map_label = nn.Linear(label_dim, noise_channels)
        elif label_dim != 0 and label_type == 'downlink':
            self.map_label = nn.Sequential(
                nn.Linear(label_dim, noise_channels * 2),
                nn.SiLU(),
                nn.Linear(noise_channels * 2, noise_channels * 2),
            )
        elif label_dim != 0 and label_type == 'snr':
            self.map_label = nn.Sequential(
                nn.Linear(label_dim, emb_channels),
                nn.SiLU(),
                nn.Linear(emb_channels, emb_channels),
            )
        else:
            self.map_label = None

        self.map_augment = nn.Linear(augment_dim, noise_channels, bias=False) if augment_dim else None

        in_emb = noise_channels * (3 if (self.map_label is not None and label_type == 'downlink') else 1)
        self.map_layer0 = nn.Linear(in_emb, emb_channels * 2)
        self.map_layer1 = nn.Linear(emb_channels * 2, emb_channels)

        # Spatial resolution tracking
        H_res = img_resolution[0] if isinstance(img_resolution, (list, tuple)) else img_resolution
        W_res = img_resolution[1] if isinstance(img_resolution, (list, tuple)) else img_resolution
        max_t = 1 << int(np.ceil(np.log2(max(1, H_res))))
        max_c = 1 << int(np.ceil(np.log2(max(1, W_res))))

        cur_t, cur_c = max_t, max_c
        down_strides = []
        for level in range(len(encdec_channel_mult) - 1):
            s_t = 2 if cur_t > 1 else 1
            s_c = 2 if cur_c > 1 else 1
            down_strides.append((s_t, s_c))
            cur_t = max(1, cur_t // s_t)
            cur_c = max(1, cur_c // s_c)

        # Stem convolution
        self.stem = ComplexConv2d(in_channels, model_channels, kernel_size=3, causal=causal_conv)

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
                    ComplexUNetBlock(
                        cin, c_target, emb_channels,
                        causal_conv=causal_conv, dropout=dropout
                    )
                )
            self.enc_blocks.append(stage_blocks)
            c_curr = c_target

            if level < len(encdec_channel_mult) - 1:
                d_stride = down_strides[level]
                self.downsamplers.append(ComplexResample2d(c_curr, down=True, causal=causal_conv, resample_filter=resample_filter, resample_stride=d_stride))

        c_bottleneck = c_curr

        # Tokenization & DiT projection
        self.token_proj = cm.ComplexLinear(c_bottleneck, dit_hidden_dim)
        self.pos_embed = ComplexPositionalEmbedding2D(pos_embed_type, dit_hidden_dim, max_t=max_t, max_c=max_c)

        # Projection from real emb to complex conditioning for DiT adaLN-Zero
        self.c_proj = nn.Linear(emb_channels, dit_hidden_dim * 2)

        # DiT Blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(dit_hidden_dim, dit_num_heads, dropout=dropout, mlp_ratio=dit_mlp_ratio)
            for _ in range(num_dit_blocks)
        ])
        self.final_layer = FinalLayer(dit_hidden_dim, c_bottleneck)

        # Decoder stages
        self.dec_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        # Mirror levels in reverse
        up_strides = list(reversed(down_strides))
        up_idx = 0
        for level in reversed(range(len(encdec_channel_mult))):
            mult = encdec_channel_mult[level]
            c_stage = model_channels * mult

            if level < len(encdec_channel_mult) - 1:
                u_stride = up_strides[up_idx]
                self.upsamplers.append(ComplexResample2d(c_curr, up=True, causal=causal_conv, resample_filter=resample_filter, resample_stride=u_stride))
                up_idx += 1

            # Concatenation of skip: input channels = c_curr + c_stage
            cin_first = c_curr + c_stage
            stage_blocks = nn.ModuleList()
            for b_idx in range(num_blocks_per_stage):
                cin = cin_first if b_idx == 0 else c_stage
                stage_blocks.append(
                    ComplexUNetBlock(
                        cin, c_stage, emb_channels,
                        causal_conv=causal_conv, dropout=dropout
                    )
                )
            self.dec_blocks.append(stage_blocks)
            c_curr = c_stage

        # Final output projection
        self.out_conv = ComplexConv2d(c_curr, out_channels, kernel_size=3, causal=causal_conv)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor, class_labels=None, augment_labels=None, z=None, **kwargs):
        """
        Args:
            x: Complex tensor [B, A, T, C] (torch.complex64) or float tensor [B, A, T, C, 2]
            noise_labels: Diffusion noise level [B] or spatial noise [B, ...]
            class_labels: Class labels or condition tensor
        """
        orig_ndim = x.ndim
        is_complex_input = torch.is_complex(x)
        is_concat_real_imag = False
        is_leading_complex = False
        if is_complex_input:
            x = torch.stack([x.real, x.imag], dim=-1)
        elif x.ndim == 5:
            if x.shape[-1] == 2 and x.shape[1] == self.in_channels:
                pass
            elif x.shape[1] == 2 and x.shape[2] == self.in_channels:
                is_leading_complex = True
                x = x.permute(0, 2, 3, 4, 1)
        elif x.ndim == 4:
            if x.shape[1] == 2 * self.in_channels:
                # 4D tensor where real and imag are concatenated along antenna dimension [B, 2*A, T, C]
                is_concat_real_imag = True
                x = torch.stack([x[:, :self.in_channels], x[:, self.in_channels:]], dim=-1)
            else:
                # 4D float tensor with in_channels, treat as real-only with zero imaginary part
                x = torch.stack([x, torch.zeros_like(x)], dim=-1)

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
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos

        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels.to(torch.float32))

        snr_emb = None
        if self.map_label is not None:
            tmp = class_labels
            if (tmp is None or (isinstance(tmp, torch.Tensor) and tmp.ndim >= 2 and tmp.shape[-1] == 0)) and z is not None:
                tmp = z
            if tmp is not None:
                tmp = tmp.to(torch.float32)
                if tmp.ndim == 1:
                    tmp = tmp.unsqueeze(-1)
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
                elif self.label_type == 'snr':
                    snr_emb = self.map_label(tmp)

        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))
        if snr_emb is not None:
            emb = emb + snr_emb

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
        B, C_enc, t_down, c_down, _ = x.shape
        N = t_down * c_down

        x_tokens = x.permute(0, 2, 3, 1, 4).reshape(B, N, C_enc, 2)
        x_tokens = self.token_proj(x_tokens)
        x_tokens = self.pos_embed(x_tokens, t_down, c_down)

        # DiT Complex conditioning
        c_dit = self.c_proj(emb).to(x.dtype).view(B, self.dit_hidden_dim, 2)

        for dit_block in self.dit_blocks:
            x_tokens = dit_block(x_tokens, c_dit)

        x_tokens = self.final_layer(x_tokens, c_dit)

        # Unpatchify back to spatial grid
        x = x_tokens.reshape(B, t_down, c_down, C_enc, 2).permute(0, 3, 1, 2, 4)

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

        if is_complex_input:
            out_r, out_i = out.unbind(-1)
            return torch.complex(out_r, out_i)
        if is_leading_complex:
            return out.permute(0, 4, 1, 2, 3)
        if is_concat_real_imag:
            return torch.cat(out.unbind(-1), dim=1)
        if orig_ndim == 4:
            return out[..., 0]
        return out
