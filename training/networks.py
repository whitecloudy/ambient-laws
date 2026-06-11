# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu
from huggingface_hub import PyTorchModelHubMixin


#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Helpers for padding to power of two.

def pad_to_power_of_two(x):
    orig_H, orig_W = x.shape[-2], x.shape[-1]
    target_H = 1 << (orig_H - 1).bit_length() if orig_H > 0 else 0
    target_W = 1 << (orig_W - 1).bit_length() if orig_W > 0 else 0
    
    pad_h = target_H - orig_H
    pad_w = target_W - orig_W
    
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
    return x, pad_h, pad_w, orig_H, orig_W

def unpad_from_power_of_two(x, pad_h, pad_w, orig_H, orig_W):
    if pad_h > 0 or pad_w > 0:
        return x[..., :orig_H, :orig_W]
    return x

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], resample_stride=2, fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0, stride=(1,1)
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        if isinstance(resample_stride, int):
            self.resample_stride = (resample_stride, resample_stride)
        else:
            self.resample_stride = tuple(resample_stride)
        
        if isinstance(kernel, int):
            kernel = [kernel, kernel]
        else:
            kernel = list(kernel)
            
        if kernel[0] == 0 and kernel[1] == 0 and (self.stride[0] > 1 or self.stride[1] > 1):
            kernel = [1, 1]
            
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel[0]*kernel[1], fan_out=out_channels*kernel[0]*kernel[1])
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel[0], kernel[1]], **init_kwargs) * init_weight) if kernel[0] > 0 and kernel[1] > 0 else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel[0] > 0 and kernel[1] > 0 and bias else None
        if up or down:
            if resample_filter is None:
                self.register_buffer('resample_filter', None)
            elif isinstance(resample_filter, (list, tuple)) and len(resample_filter) == 2 and isinstance(resample_filter[0], (list, tuple)):
                f_h = torch.as_tensor(resample_filter[0], dtype=torch.float32)
                f_w = torch.as_tensor(resample_filter[1], dtype=torch.float32)
            else:
                f_h = torch.as_tensor(resample_filter, dtype=torch.float32)
                f_w = f_h
            if resample_filter is not None:
                f = f_h.ger(f_w)
                f = f.unsqueeze(0).unsqueeze(1) / (f_h.sum() * f_w.sum())
                self.register_buffer('resample_filter', f)
        else:
            self.register_buffer('resample_filter', None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = (max(0, w.shape[-2] - self.stride[0] + 1) // 2, max(0, w.shape[-1] - self.stride[1] + 1) // 2) if w is not None else (0, 0)
        f_pad = ((f.shape[-2] - 1) // 2, (f.shape[-1] - 1) // 2) if f is not None else (0, 0)
        stride_mul = self.resample_stride[0] * self.resample_stride[1]
        
        out_pad = (0, 0)
        if f is not None:
            out_pad = ((2 * f_pad[0] - f.shape[-2]) % self.resample_stride[0],
                       (2 * f_pad[1] - f.shape[-1]) % self.resample_stride[1])
        
        def w_out_pad_fn(pad_val):
            if w is None: return (0, 0)
            return ((2 * pad_val[0] - w.shape[-2]) % self.stride[0],
                    (2 * pad_val[1] - w.shape[-1]) % self.stride[1])

        if self.fused_resample and self.up and w is not None and f is not None:
            pad_t = (max(f_pad[0] - w_pad[0], 0), max(f_pad[1] - w_pad[1], 0))
            x = torch.nn.functional.conv_transpose2d(x, f.mul(stride_mul).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=self.resample_stride, padding=pad_t, output_padding=out_pad)
            pad_c = (max(w_pad[0] - f_pad[0], 0), max(w_pad[1] - f_pad[1], 0))
            if self.stride[0] > 1 or self.stride[1] > 1:
                x = torch.nn.functional.conv_transpose2d(x, w.transpose(0, 1), padding=pad_c, output_padding=w_out_pad_fn(pad_c), stride=self.stride)
            else:
                x = torch.nn.functional.conv2d(x, w, padding=pad_c, stride=self.stride)
        elif self.fused_resample and self.down and w is not None and f is not None:
            pad_c = (w_pad[0] + f_pad[0], w_pad[1] + f_pad[1])
            x = torch.nn.functional.conv2d(x, w, padding=pad_c, stride=self.stride)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=self.resample_stride)
        else:
            if self.up and f is not None:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(stride_mul).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=self.resample_stride, padding=f_pad, output_padding=out_pad)
            if self.down and f is not None:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=self.resample_stride, padding=f_pad)
            if w is not None:
                if self.up and (self.stride[0] > 1 or self.stride[1] > 1):
                    x = torch.nn.functional.conv_transpose2d(x, w.transpose(0, 1), padding=w_pad, output_padding=w_out_pad_fn(w_pad), stride=self.stride)
                else:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad, stride=self.stride)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_stride=2, resample_proj=False, adaptive_scale=True,
        kernel=3,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None, stride=(1,1),
        emb_stride=(1,1) # 추가: emb 전용 stride
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        if isinstance(stride, int):
            stride = (stride, stride)
        else:
            stride = tuple(stride)

        if not (up or down):
            stride = (1, 1)

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, resample_stride=resample_stride, stride=stride, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=kernel, **init_zero)

        self.skip = None
        has_stride = stride[0] > 1 or stride[1] > 1
        if out_channels != in_channels or up or down or has_stride:
            skip_kernel = 1 if resample_proj or out_channels != in_channels or has_stride else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=skip_kernel, up=up, down=down, resample_filter=resample_filter, resample_stride=resample_stride, stride=stride, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x
    
#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock_AS(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_stride=2, resample_proj=False, adaptive_scale=True,
        kernel=3,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None, stride=(1,1),
        emb_stride=(1,1) # 유지 호환성을 위해 추가
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        # emb를 위한 Learnable Downsampling Layer 추가
        self.emb_stride = emb_stride if isinstance(emb_stride, (tuple, list)) else (emb_stride, emb_stride)
        if self.emb_stride[0] > 1 or self.emb_stride[1] > 1:
            self.emb_down = Conv2d(in_channels=emb_channels, out_channels=emb_channels, kernel=self.emb_stride, stride=self.emb_stride, **init)
        else:
            self.emb_down = torch.nn.Identity()

        if isinstance(stride, int):
            stride = (stride, stride)
        else:
            stride = tuple(stride)

        if not (up or down):
            stride = (1, 1)

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, resample_stride=resample_stride, stride=stride, **init)
        self.affine = Conv2d(in_channels=emb_channels, out_channels=out_channels*(2 if adaptive_scale else 1), kernel=1, **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=kernel, **init_zero)

        self.skip = None
        has_stride = stride[0] > 1 or stride[1] > 1
        if out_channels != in_channels or up or down or has_stride:
            skip_kernel = 1 if resample_proj or out_channels != in_channels or has_stride else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=skip_kernel, up=up, down=down, resample_filter=resample_filter, resample_stride=resample_stride, stride=stride, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        # Conv Layer + Stride를 이용한 emb Downsampling 수행
        emb_input = self.emb_down(emb)
        if emb_input.shape[2:] != x.shape[2:]:  # 아주 미세한 반올림 오차 대비용 안전 장치
            emb_input = torch.nn.functional.interpolate(emb_input, size=x.shape[2:], mode='nearest')

        params = self.affine(emb_input).to(x.dtype)

        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x) + params)

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embeddings with Spatial mapping support

@persistence.persistent_class
class PositionalEmbedding_AS(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        
        if x.ndim >= 3:
            if x.ndim == 4:
                # x: [B, C, H, W]
                emb = x.unsqueeze(-1) * freqs.to(x.dtype) # [B, C, H, W, F]
                emb = torch.cat([emb.cos(), emb.sin()], dim=-1) # [B, C, H, W, 2F]
                emb = emb.permute(0, 4, 1, 2, 3) # [B, 2F, C, H, W]
                x = torch.mean(emb, dim=2) # [B, 2F, H, W]
            else: # x.ndim == 3, i.e., [B, H, W]
                x = x.unsqueeze(-1) * freqs.to(x.dtype)
                x = torch.cat([x.cos(), x.sin()], dim=-1)
                x = x.permute(0, 3, 1, 2)
        else:
            x = x.ger(freqs.to(x.dtype))
            x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

@persistence.persistent_class
class FourierEmbedding_AS(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        if x.ndim >= 3:
            freqs = (2 * np.pi * self.freqs).to(x.dtype)
            if x.ndim == 4:
                # 각 채널별로 Fourier Encoding을 적용한 후, 채널 축에 대해 평균을 계산합니다.
                # x: [B, C, H, W]
                emb = x.unsqueeze(-1) * freqs # [B, C, H, W, F]
                emb = torch.cat([emb.cos(), emb.sin()], dim=-1) # [B, C, H, W, 2F]
                emb = emb.permute(0, 4, 1, 2, 3) # [B, 2F, C, H, W]
                x = torch.mean(emb, dim=2) # [B, 2F, H, W]
            else: # x.ndim == 3, i.e., [B, H, W]
                x = x.unsqueeze(-1) * freqs
                x = torch.cat([x.cos(), x.sin()], dim=-1)
                x = x.permute(0, 3, 1, 2)
        else:
            x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
            x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------

# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@persistence.persistent_class
class SongUNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux
    

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# Modifeid for RF data. Minor changes to the original implementation
# available at https://github.com/yang-song/score_sde_pytorch

from collections import OrderedDict

@persistence.persistent_class
class RF_SongUNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        label_resolution    = None,         # Label resolution
        label_type          = 'downlink',
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        dynamic_noise       = False,        # Whether to use dynamic noise labels.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']
        assert label_type in ['downlink', 'classes', 'no_label']

        super().__init__()
        self.label_dropout = label_dropout
        self.label_type = label_type

        # dynamic_noise일 경우 Spatial Tensor가 되어 VRAM 소모가 매우 커지므로 기본 embedding 차원을 줄입니다.
        if dynamic_noise and channel_mult_emb == 4:
            channel_mult_emb = 1

        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        self.dynamic_noise = dynamic_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        if self.dynamic_noise:
            self.map_noise = PositionalEmbedding_AS(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding_AS(num_channels=noise_channels)
        else:
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        if label_dim != 0:
            if self.label_type == 'downlink':
                if label_resolution == None:
                    label_resolution = img_resolution
                # Apply padding size
                padded_label_H = 1 << (label_resolution[0] - 1).bit_length() if label_resolution[0] > 0 else 0
                padded_label_W = 1 << (label_resolution[1] - 1).bit_length() if label_resolution[1] > 0 else 0
                
                flatten_feature_size = padded_label_H//4 * padded_label_W//4 * model_channels
                self.map_label = torch.nn.Sequential(
                    (OrderedDict([
                                ("Label encoder", Conv2d(in_channels=label_dim, out_channels=model_channels, kernel=3, **init)),    # RT : (model_channels, padded_label_H, padded_label_W)
                                ("SiLU 1", torch.nn.SiLU()),
                                ("GroupNorm 1", GroupNorm(num_channels=model_channels, eps=1e-6)),
                                ("Label UNet 1", Conv2d(in_channels=model_channels, out_channels=model_channels, kernel=3, down=True, **init)),    # RT : (model_channels, padded_label_H//2, padded_label_W//2)
                                ("SiLU 2", torch.nn.SiLU()),
                                ("GroupNorm 2", GroupNorm(num_channels=model_channels, eps=1e-6)),
                                ("Label UNet 2", Conv2d(in_channels=model_channels, out_channels=model_channels, kernel=3, down=True, **init)),    # RT : (model_channels, padded_label_H//4, padded_label_W//4)
                                ("Flatten", torch.nn.Flatten()),    # RT : (model_channels*2 * padded_label_H//4 * padded_label_W//4)
                                ("Linear embedding", Linear(in_features=flatten_feature_size, out_features=noise_channels*2, **init)),
                                ("Final Layer Norm", torch.nn.LayerNorm(noise_channels*2)),
                    ]))
                )
            elif self.label_type == 'classes':
                self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init)
            elif self.label_type == 'no_label':
                self.map_label = None
            else:  
                assert False, "Unknown label type"
        else:
            self.map_label = None

        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None

        if self.map_label != None and self.label_type == 'downlink':
            if self.dynamic_noise:
                self.map_layer0 = Conv2d(in_channels=noise_channels*3, out_channels=emb_channels*2, kernel=1, **init)
                self.map_layer1 = Conv2d(in_channels=emb_channels*2, out_channels=emb_channels, kernel=1, **init)
            else:
                self.map_layer0 = Linear(in_features=noise_channels*3, out_features=emb_channels*2, **init)
                self.map_layer1 = Linear(in_features=emb_channels*2, out_features=emb_channels, **init)
        else:
            if self.dynamic_noise:
                self.map_layer0 = Conv2d(in_channels=noise_channels, out_channels=emb_channels, kernel=1, **init)
                self.map_layer1 = Conv2d(in_channels=emb_channels, out_channels=emb_channels, kernel=1, **init)
            else:
                self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
                self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        unet_block_class = UNetBlock_AS if self.dynamic_noise else UNetBlock

        for level, mult in enumerate(channel_mult):
            H_res = img_resolution[0] >> level
            W_res = img_resolution[1] >> level
            emb_stride = (img_resolution[0] // max(H_res, 1), img_resolution[1] // max(W_res, 1))
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{H_res}x{W_res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{H_res}x{W_res}_down'] = unet_block_class(in_channels=cout, out_channels=cout, down=True, emb_stride=emb_stride, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{H_res}x{W_res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{H_res}x{W_res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{H_res}x{W_res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (W_res in attn_resolutions)
                self.enc[f'{H_res}x{W_res}_block{idx}'] = unet_block_class(in_channels=cin, out_channels=cout, attention=attn, emb_stride=emb_stride, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            H_res = img_resolution[0] >> level
            W_res = img_resolution[1] >> level            
            emb_stride = (img_resolution[0] // max(H_res, 1), img_resolution[1] // max(W_res, 1))
            if level == len(channel_mult) - 1:
                self.dec[f'{H_res}x{W_res}_in0'] = unet_block_class(in_channels=cout, out_channels=cout, attention=True, emb_stride=emb_stride, **block_kwargs)
                self.dec[f'{H_res}x{W_res}_in1'] = unet_block_class(in_channels=cout, out_channels=cout, emb_stride=emb_stride, **block_kwargs)
            else:
                self.dec[f'{H_res}x{W_res}_up'] = unet_block_class(in_channels=cout, out_channels=cout, up=True, emb_stride=emb_stride, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and W_res in attn_resolutions)
                self.dec[f'{H_res}x{W_res}_block{idx}'] = unet_block_class(in_channels=cin, out_channels=cout, attention=attn, emb_stride=emb_stride, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{H_res}x{W_res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{H_res}x{W_res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{H_res}x{W_res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # [Shape 가정] x: [B, C, H, W], noise_labels: [B, C, H, W], class_labels: [B, K]
        
        x, pad_h, pad_w, orig_H, orig_W = pad_to_power_of_two(x)
        if self.label_type == 'downlink':
            class_labels, c_pad_h, c_pad_w, c_orig_H, c_orig_W = pad_to_power_of_two(class_labels)

            if pad_h != c_pad_h or pad_w != c_pad_w:
                print("WARNING, Input class and x have different shape")
        
        is_spatial_noise = (noise_labels.ndim >= 3 and noise_labels.shape[-2:] == (orig_H, orig_W))
        
        if self.dynamic_noise and is_spatial_noise and (pad_h > 0 or pad_w > 0):
            noise_labels = torch.nn.functional.pad(noise_labels, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        if self.dynamic_noise:
            while noise_labels.ndim < x.ndim:
                noise_labels = noise_labels.unsqueeze(-1)
                
            if noise_labels.shape != x.shape:
                noise_labels = noise_labels.expand_as(x)
            # dynamic_noise=True 일 때 noise_labels 최종 shape: [B, C, H, W]
        else:   #If not dynamic noise but when we receive dynamic noise label
            if noise_labels.ndim > 1:   # Expect only [Batch, ]
                noise_labels = torch.sqrt(torch.mean(torch.flatten(noise_labels, start_dim=1)**2, dim=1))
            # dynamic_noise=False 일 때 noise_labels 최종 shape: [B]

        # Mapping.    
        emb = self.map_noise(noise_labels)
        # dynamic_noise=True 일 때 emb shape: [B, noise_channels, H, W]
        # dynamic_noise=False 일 때 emb shape: [B, noise_channels]
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_augment is not None and augment_labels is not None:
            augment_emb = self.map_augment(augment_labels)
            if self.dynamic_noise:
                augment_emb = augment_emb.unsqueeze(-1).unsqueeze(-1)
            emb = emb + augment_emb
        if self.map_label is not None:
            tmp = class_labels
            # tmp shape: [B, K]
            if self.training and self.label_dropout and self.label_type != 'no_label':
                label_dropout_table = torch.unsqueeze(torch.unsqueeze((torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype), dim=-1), dim=-1)
                tmp = tmp * label_dropout_table
                # label_dropout 적용 후 tmp shape: [B, K]
                
            if self.label_type == 'downlink':
                label_emb = self.map_label(tmp)
                # label_emb shape: [B, noise_channels * 2] (map_label 설계에 따름)
                if self.dynamic_noise:
                    label_emb = label_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, emb.shape[2], emb.shape[3])
                    # label_emb shape: [B, noise_channels * 2, H, W]
                emb = torch.concatenate([emb, label_emb], dim=1)
                # emb shape (dynamic_noise=True): [B, noise_channels * 3, H, W]
                # emb shape (dynamic_noise=False): [B, noise_channels * 3]
            elif self.label_type == 'classes':
                label_emb = self.map_label(tmp * np.sqrt(self.map_label.in_features))
                # label_emb shape: [B, noise_channels]
                if self.dynamic_noise:
                    label_emb = label_emb.unsqueeze(-1).unsqueeze(-1)
                    # label_emb shape: [B, noise_channels, 1, 1]
                emb = emb + label_emb
                # emb shape (dynamic_noise=True): [B, noise_channels, H, W]
                # emb shape (dynamic_noise=False): [B, noise_channels]
            elif self.label_type == 'no_label':
                pass
            else:
                assert False, "Unknown label type"
        emb = silu(self.map_layer0(emb))
        # emb shape (label_type='downlink', dynamic_noise=True): [B, emb_channels * 2, H, W]
        # emb shape (label_type='downlink', dynamic_noise=False): [B, emb_channels * 2]
        # emb shape (label_type='classes', dynamic_noise=True): [B, emb_channels, H, W]
        # emb shape (label_type='classes', dynamic_noise=False): [B, emb_channels]
        emb = silu(self.map_layer1(emb))
        # emb 최종 shape (dynamic_noise=True): [B, emb_channels, H, W]
        # emb 최종 shape (dynamic_noise=False): [B, emb_channels]

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if (isinstance(block, UNetBlock) or isinstance(block, UNetBlock_AS)) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                    
                x = block(x, emb)

        aux = unpad_from_power_of_two(aux, pad_h, pad_w, orig_H, orig_W)

        return aux


@persistence.persistent_class
class WiDAR_RF_SongUNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        label_resolution    = None,         # Label resolution
        label_type          = 'classes',
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        resample_stride     = [2,2],        
        kernel_size         = [3,3],       # Base kernel size.
        stride              = [1,1],       # Base stride.
        stem_kernel         = [3,3],       # Initial stem kernel size (Patchification)
        stem_stride         = [1,1]        # Initial stem downsampling stride (Patchification)
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']
        assert label_type in ['sigma', 'both', 'classes']

        super().__init__()
        self.label_dropout = label_dropout
        self.label_type = label_type
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_stride=resample_stride, resample_proj=True, adaptive_scale=False,
            kernel=kernel_size, stride=stride,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )
        
        if isinstance(stem_stride, int):
            self.stem_stride = (stem_stride, stem_stride)
        else:
            self.stem_stride = tuple(stem_stride)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        if label_dim != 0:
            if label_type == 'sigma':
                if label_resolution == None:
                    label_resolution = img_resolution
                
                padded_label_H = 1 << (label_resolution[0] - 1).bit_length() if label_resolution[0] > 0 else 0
                padded_label_W = 1 << (label_resolution[1] - 1).bit_length() if label_resolution[1] > 0 else 0

                total_stride_0 = resample_stride[0] * stride[0]
                total_stride_1 = resample_stride[1] * stride[1]
                flatten_feature_size = (padded_label_H // (total_stride_0**2)) * (padded_label_W // (total_stride_1**2)) * model_channels
                self.map_label = torch.nn.Sequential(
                    (OrderedDict([
                                ("Label encoder", Conv2d(in_channels=label_dim, out_channels=model_channels, kernel=kernel_size, **init)),    # RT : (model_channels, padded_label_H, padded_label_W)
                                ("SiLU 1", torch.nn.SiLU()),
                                ("GroupNorm 1", GroupNorm(num_channels=model_channels, eps=1e-6)),
                                ("Label UNet 1", Conv2d(in_channels=model_channels, out_channels=model_channels, kernel=kernel_size, down=True, resample_stride=resample_stride, stride=stride, **init)),    # RT : (model_channels, padded_label_H//2, padded_label_W//2)
                                ("SiLU 2", torch.nn.SiLU()),
                                ("GroupNorm 2", GroupNorm(num_channels=model_channels, eps=1e-6)),
                                ("Label UNet 2", Conv2d(in_channels=model_channels, out_channels=model_channels, kernel=kernel_size, down=True, resample_stride=resample_stride, stride=stride, **init)),    # RT : (model_channels, padded_label_H//4, padded_label_W//4)
                                ("Flatten", torch.nn.Flatten()),    # RT : (model_channels*2 * padded_label_H//4 * padded_label_W//4)
                                ("Linear embedding", Linear(in_features=flatten_feature_size, out_features=noise_channels*2, **init)),
                                ("Final Layer Norm", torch.nn.LayerNorm(noise_channels*2)),
                    ]))
                )
            elif label_type == 'classes':
                self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init)
            elif label_type == 'both':
                if label_resolution == None:
                    label_resolution = img_resolution
                
                padded_label_H = 1 << (label_resolution[0] - 1).bit_length() if label_resolution[0] > 0 else 0
                padded_label_W = 1 << (label_resolution[1] - 1).bit_length() if label_resolution[1] > 0 else 0

                total_stride_0 = resample_stride[0] * stride[0]
                total_stride_1 = resample_stride[1] * stride[1]
                flatten_feature_size = (padded_label_H // (total_stride_0**2)) * (padded_label_W // (total_stride_1**2)) * model_channels
                self.map_label = torch.nn.ModuleList([torch.nn.Sequential(
                    (OrderedDict([
                                ("Label encoder", Conv2d(in_channels=label_dim, out_channels=model_channels, kernel=kernel_size, **init)),    # RT : (model_channels, padded_label_H, padded_label_W)
                                ("SiLU 1", torch.nn.SiLU()),
                                ("GroupNorm 1", GroupNorm(num_channels=model_channels, eps=1e-6)),
                                ("Label UNet 1", Conv2d(in_channels=model_channels, out_channels=model_channels, kernel=kernel_size, down=True, resample_stride=resample_stride, stride=stride, **init)),    # RT : (model_channels, padded_label_H//2, padded_label_W//2)
                                ("SiLU 2", torch.nn.SiLU()),
                                ("GroupNorm 2", GroupNorm(num_channels=model_channels, eps=1e-6)),
                                ("Label UNet 2", Conv2d(in_channels=model_channels, out_channels=model_channels, kernel=kernel_size, down=True, resample_stride=resample_stride, stride=stride, **init)),    # RT : (model_channels, padded_label_H//4, padded_label_W//4)
                                ("Flatten", torch.nn.Flatten()),    # RT : (model_channels*2 * padded_label_H//4 * padded_label_W//4)
                                ("Linear embedding", Linear(in_features=flatten_feature_size, out_features=noise_channels*2, **init)),
                                ("Final Layer Norm", torch.nn.LayerNorm(noise_channels*2)),
                    ]))
                ), Linear(in_features=label_dim, out_features=noise_channels, **init)])
            else:  
                assert False, "Unknown label type"
        else:
            self.map_label = None

        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None

        if self.map_label != None and self.label_type == 'downlink':
            self.map_layer0 = Linear(in_features=noise_channels*3, out_features=emb_channels*2, **init)
            self.map_layer1 = Linear(in_features=emb_channels*2, out_features=emb_channels, **init)
        else:
            self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        cur_H = img_resolution[0]
        cur_W = img_resolution[1]
        level_resolutions = []

        for level, mult in enumerate(channel_mult):
            level_resolutions.append((cur_H, cur_W))
            if level == 0:
                cin = cout
                cout = model_channels
                if self.stem_stride[0] > 1 or self.stem_stride[1] > 1:
                    self.enc[f'{cur_H}x{cur_W}_conv_stem'] = Conv2d(in_channels=cin, out_channels=cout, kernel=stem_kernel, stride=self.stem_stride, **init)
                else:
                    self.enc[f'{cur_H}x{cur_W}_conv_stem'] = Conv2d(in_channels=cin, out_channels=cout, kernel=stem_kernel, **init)
                cur_H = cur_H // self.stem_stride[0]
                cur_W = cur_W // self.stem_stride[1]
            else:
                self.enc[f'{cur_H}x{cur_W}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{cur_H}x{cur_W}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter, resample_stride=resample_stride, stride=stride)
                    self.enc[f'{cur_H}x{cur_W}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{cur_H}x{cur_W}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=kernel_size, down=True, resample_filter=resample_filter, resample_stride=resample_stride, fused_resample=True, stride=stride, **init)
                    caux = cout
                cur_H = cur_H // (resample_stride[0] * stride[0])
                cur_W = cur_W // (resample_stride[1] * stride[1])

            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (cur_W in attn_resolutions)
                self.enc[f'{cur_H}x{cur_W}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            cur_H, cur_W = level_resolutions[level]
            if level == len(channel_mult) - 1:
                self.dec[f'{cur_H}x{cur_W}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{cur_H}x{cur_W}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{cur_H}x{cur_W}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and cur_W in attn_resolutions)
                self.dec[f'{cur_H}x{cur_W}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{cur_H}x{cur_W}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter, resample_stride=resample_stride, stride=stride)
                self.dec[f'{cur_H}x{cur_W}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                if self.stem_stride[0] > 1 or self.stem_stride[1] > 1:
                    self.dec[f'{cur_H}x{cur_W}_aux_conv_stem'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=stem_kernel, up=True, resample_filter=None, stride=self.stem_stride, **init_zero)
                else:
                    self.dec[f'{cur_H}x{cur_W}_aux_conv_stem'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=stem_kernel, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        x, pad_h, pad_w, orig_H, orig_W = pad_to_power_of_two(x)

        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                label_dropout_table = torch.unsqueeze(torch.unsqueeze((torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype), dim=-1), dim=-1)

                tmp = tmp * label_dropout_table
            if self.label_type == 'sigma':
                if pad_h > 0 or pad_w > 0:
                    tmp = torch.nn.functional.pad(tmp, (0, pad_w, 0, pad_h), mode='constant', value=0)
                label_emb = self.map_label(tmp)
                # emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
                emb = torch.concatenate([emb, label_emb], dim=1)
            elif self.label_type == 'classes':
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            elif self.label_type == 'both':
                tmp_0 = tmp[0]
                if pad_h > 0 or pad_w > 0:
                    tmp_0 = torch.nn.functional.pad(tmp_0, (0, pad_w, 0, pad_h), mode='constant', value=0)
                emb = emb + self.map_label[1](tmp[1] * np.sqrt(self.map_label[1].in_features))
                label_emb = self.map_label[0](tmp_0)
                emb = torch.concatenate([emb, label_emb], dim=1)
            else:
                assert False, "Unknown label type"
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)

        aux = unpad_from_power_of_two(aux, pad_h, pad_w, orig_H, orig_W)

        return aux

from training.wifi_model import tfdiff_WiFi

class RF_transformer(tfdiff_WiFi):
    def __init__(self, img_resolution, in_channels, label_dim, label_type, out_channels, *args, **kwargs):
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.label_dim = label_dim
        self.label_type = label_type
        self.out_channels = out_channels
        super().__init__(*args, **kwargs)


#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        beta_d          = 19.9,         # Extent of the noise level schedule.
        beta_min        = 0.1,          # Initial slope of the noise level schedule.
        M               = 1000,         # Original number of timesteps in the DDPM formulation.
        epsilon_t       = 1e-5,         # Minimum t-value used during training.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        sigma_min       = 0.02,         # Minimum supported noise level.
        sigma_max       = 100,          # Maximum supported noise level.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x : torch.Tensor, sigma : torch.Tensor, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)
        
        while sigma.ndim < x.ndim:
            sigma = sigma.unsqueeze(-1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        
        safe_sigma = torch.where(sigma == 0.0, torch.tensor(1e-8, dtype=sigma.dtype, device=sigma.device), sigma)
        c_noise = safe_sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise, class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def generate(self, batch_size=1, device="cuda",
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    ):
        net = self.to(device)
        # Pick latents and labels.
        latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:]))): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_next
#----------------------------------------------------------------------------
