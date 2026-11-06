#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_SC_stage3.py - SC-CAD2M Stage 3 Joint Fine-tuning

Paper: "Toward Reliable Semantic Communications: A Fast One-Step Channel-Adaptive Denoising Diffusion Method"
IEEE Transactions on Networking (2026)

Stage 3 Joint Fine-Tuning Pipeline:
1) Clean image s is passed through the adaJSCC Encoder f_omega(s, zeta) to generate latent z.
2) Latent z passes through the physical wireless channel and noise is added to produce received signal y.
3) Received signal y undergoes channel-adaptive diffusion denoising sampling (CAD2M) to produce denoised latent x
   (with channel equalization if applicable).
4) Denoised latent x is passed through the adaJSCC Decoder f_phi(x, zeta) to reconstruct image s_hat.
5) All modules (Encoder, Diffusion U-Net, Decoder) or selected subsets are fine-tuned end-to-end to minimize
   distortion L3(omega, phi, theta) = E[ ||s - s_hat||^2 ].
"""

import os
import sys
import re
import json
import time
import copy
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure

import dnnlib
from torch_utils import distributed as dist
from torch_utils import save_utils
from training.sampler import inference_edm_sampler as edm_sampler
from training.sampler import padding_mask_from_original_shape
from training.semantic_comm_dataset import (
    _load_adjscc_config,
    find_adjscc_checkpoint,
    calculate_sc_t_from_config,
    RenewChannelDataset,
)

# Ensure misc/ADJSCC is in sys.path
_WORKSPACE_DIR = Path(__file__).resolve().parent
_ADJSCC_DIR = _WORKSPACE_DIR / 'misc' / 'ADJSCC'
if str(_ADJSCC_DIR) not in sys.path:
    sys.path.insert(0, str(_ADJSCC_DIR))

from misc.ADJSCC.util_module import (
    Attention_Encoder,
    Attention_Decoder,
    Attention_Encoder_H,
    Attention_Decoder_H,
)


def _power2ceil(original: int) -> int:
    return int(2 ** np.ceil(np.log2(original)))


def complex_nan_to_num(z: torch.Tensor, nan: float = 0.0, posinf: float = 10.0, neginf: float = -10.0) -> torch.Tensor:
    """복소수 텐서의 autograd(자동 미분)를 지원하는 nan_to_num 대체 함수."""
    if torch.is_complex(z):
        r = torch.nan_to_num(z.real, nan=nan, posinf=posinf, neginf=neginf)
        i = torch.nan_to_num(z.imag, nan=nan, posinf=posinf, neginf=neginf)
        return torch.complex(r, i)
    return torch.nan_to_num(z, nan=nan, posinf=posinf, neginf=neginf)


def unwrap_module(m: nn.Module) -> nn.Module:
    """DDP 및 torch.compile 래퍼를 벗겨내어 원본 모듈 반환."""
    while hasattr(m, 'module') or hasattr(m, '_ddp_module'):
        if hasattr(m, '_ddp_module'):
            m = m._ddp_module
        elif hasattr(m, 'module'):
            m = m.module
    return getattr(m, '_orig_mod', m)


class DDPNetWrapper(nn.Module):
    """DistributedDataParallel wrapper that forwards attributes like sigma_max, sigma_min, round_sigma to module."""
    def __init__(self, ddp_model: nn.Module):
        super().__init__()
        self._ddp_module = ddp_model

    def forward(self, *args, **kwargs):
        return self._ddp_module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        ddp = self.__dict__.get('_modules', {}).get('_ddp_module', None)
        if ddp is None:
            ddp = self.__dict__.get('_ddp_module', None)
        if ddp is not None:
            if hasattr(ddp, name):
                return getattr(ddp, name)
            if hasattr(ddp, 'module') and hasattr(ddp.module, name):
                return getattr(ddp.module, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def pad_collate_fn(img: torch.Tensor, label: Optional[torch.Tensor] = None):
    """(N, C, H, W) 텐서의 마지막 두 차원을 2의 거듭제곱 크기로 패딩."""
    h, w = img.shape[-2], img.shape[-1]
    target_h = _power2ceil(h)
    target_w = _power2ceil(w)
    pad_h = target_h - h
    pad_w = target_w - w

    original_shape = torch.tensor(img.shape[1:], dtype=torch.long, device=img.device)
    original_shape = original_shape.unsqueeze(0).expand(img.shape[0], -1)

    if pad_h > 0 or pad_w > 0:
        pad_width = (0, pad_w, 0, pad_h)
        img = F.pad(img, pad_width, mode='constant', value=0)
        if label is not None and img.shape == label.shape:
            label = F.pad(label, pad_width, mode='constant', value=0)

    return img, label, original_shape


def load_adjscc_models(
    adjscc_config_path: str,
    explicit_model_path: Optional[str] = None,
    device: torch.device = torch.device('cuda')
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any], int, bool]:
    """ADJSCC config 및 체크포인트를 로드하고 Encoder, Decoder 인스턴스를 반환."""
    cfg = _load_adjscc_config(adjscc_config_path)
    tcn = int(cfg.get('transmit_channel_num', cfg.get('tcn', 16)))
    in_channels = int(cfg.get('in_channels', 3))

    resolved_model_path = find_adjscc_checkpoint(
        adjscc_config=cfg,
        explicit_path=explicit_model_path,
        config_file_path=Path(adjscc_config_path) if adjscc_config_path else None
    )
    if resolved_model_path is None or not Path(resolved_model_path).exists():
        raise FileNotFoundError(f"Cannot find ADJSCC model checkpoint for config: {adjscc_config_path}")

    dist.print0(f'Loading ADJSCC model from "{resolved_model_path}" (tcn={tcn})...')
    ckpt = torch.load(resolved_model_path, map_location=device)

    is_h_module = False
    for k, v in ckpt.items():
        if 'af1.fc1.weight' in k and v.shape[1] == 259:
            is_h_module = True
            break

    if is_h_module:
        dist.print0('Detected Attention_Encoder_H / Decoder_H architecture (H-conditioned).')
        encoder = Attention_Encoder_H(tcn=tcn, in_channels=in_channels)
        decoder = Attention_Decoder_H(tcn=tcn, out_channels=in_channels)
    else:
        dist.print0('Detected standard Attention_Encoder / Decoder architecture.')
        encoder = Attention_Encoder(tcn=tcn, in_channels=in_channels)
        decoder = Attention_Decoder(tcn=tcn, out_channels=in_channels)

    # Load encoder weights
    enc_dict = {k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder.')}
    if enc_dict:
        encoder.load_state_dict(enc_dict, strict=False)
    elif 'encoder' in ckpt:
        encoder.load_state_dict(ckpt['encoder'], strict=False)
    else:
        encoder.load_state_dict(ckpt, strict=False)

    # Load decoder weights
    dec_dict = {k.replace('decoder.', ''): v for k, v in ckpt.items() if k.startswith('decoder.')}
    if dec_dict:
        decoder.load_state_dict(dec_dict, strict=False)
    elif 'decoder' in ckpt:
        decoder.load_state_dict(ckpt['decoder'], strict=False)

    encoder.to(device)
    decoder.to(device)
    return encoder, decoder, cfg, tcn, is_h_module


def get_cifar10_dataset(cifar_data_root: str, is_train: bool = True) -> Dataset:
    """CIFAR-10 학습 또는 검증 데이터셋 로드."""
    transform = T.Compose([T.ToTensor()])
    dir_name = 'train' if is_train else 'test'
    candidates = [
        os.path.join(cifar_data_root, 'cifar10', dir_name),
        os.path.join(cifar_data_root, dir_name),
        os.path.join(str(_WORKSPACE_DIR), 'data', 'cifar10', dir_name),
        f'./data/cifar10/{dir_name}',
        os.path.join(str(_ADJSCC_DIR), 'data', 'cifar10', dir_name),
    ]
    for cand in candidates:
        if os.path.exists(cand) and os.path.isdir(cand):
            from torchvision.datasets import ImageFolder
            dist.print0(f'Using CIFAR-10 ({dir_name}) dataset from directory: {cand}')
            return ImageFolder(cand, transform=transform)

    dist.print0(f'Loading torchvision CIFAR-10 (train={is_train}, root={cifar_data_root})...')
    try:
        from torchvision.datasets import CIFAR10
        return CIFAR10(root=cifar_data_root, train=is_train, download=False, transform=transform)
    except Exception:
        from torchvision.datasets import CIFAR10
        return CIFAR10(root=cifar_data_root, train=is_train, download=True, transform=transform)


class PairedSCDataset(Dataset):
    """
    이미지 데이터셋과 RENEW RF 채널 데이터셋을 1:1 결정론적으로 페어링하는 데이터셋.
    RENEW 데이터셋이 없을 경우 (AWGN, Rayleigh 등 합성 채널), 더미 채널 반환.
    """
    def __init__(
        self,
        image_dataset: Dataset,
        renew_channel_dataset: Optional[Dataset] = None,
        max_samples: Optional[int] = None
    ):
        self.image_dataset = image_dataset
        self.renew_dataset = renew_channel_dataset
        self.n_images = len(image_dataset)
        self.n_channels = len(renew_channel_dataset) if renew_channel_dataset is not None else 0

        total = self.n_images
        if max_samples is not None:
            total = min(total, int(max_samples))
        self.length = total

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_img = self.image_dataset[idx]
        img = item_img[0] if isinstance(item_img, (tuple, list)) else (item_img['image'] if isinstance(item_img, dict) else item_img)
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)

        result = {
            'image': img.float(),
            'idx': idx
        }

        if self.renew_dataset is not None and self.n_channels > 0:
            ch_idx = idx % self.n_channels
            ch_item = self.renew_dataset[ch_idx]
            raw_h = ch_item.get('image', ch_item.get('clean_channel'))
            result['channel'] = raw_h.float() if isinstance(raw_h, torch.Tensor) else torch.from_numpy(raw_h).float()
        else:
            result['channel'] = torch.zeros(1, dtype=torch.float32)

        return result


def forward_stage3_pipeline(
    x_clean: torch.Tensor,
    raw_h: torch.Tensor,
    snr_tensor: torch.Tensor,
    encoder: nn.Module,
    net: nn.Module,
    decoder: nn.Module,
    is_h_module: bool,
    is_synthetic_channel: bool,
    is_rayleigh: bool,
    is_equalized: bool,
    is_awgn: bool,
    target_numel: int,
    dim_z: int,
    tcn_feat: int,
    h_feat: int,
    w_feat: int,
    denoise_steps: int = 1,
    data_norm: float = 1.0,
    padding_power2: bool = False,
    normalize_power: bool = True,
    update_encoder: bool = True,
    update_diffusion: bool = True,
    update_decoder: bool = True,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stage 3 순전파 파이프라인:
    1) Encoder: x_clean -> z_in (power normalized)
    2) Channel & Noise: z_in -> y (rx_signal)
    3) Diffusion Denoising: rx_signal -> x_denoised (1-step direct 또는 multi-step EDM)
    4) Equalization & Decoder: x_denoised -> x_hat (reconstructed image)
    
    Returns:
        x_hat: 복원된 이미지 (B, C, H, W)
        x_clean: 원본 이미지 (B, C, H, W)
        denoised_target_signal: 디퓨전 복원 복소 신호
    """
    B = x_clean.shape[0]
    device = x_clean.device

    # 1. 채널 행렬 H 변환 (Encoder 호출 전 생성하여 is_h_module에서도 정확한 채널 평균 전달)
    if torch.is_complex(raw_h):
        h_complex = raw_h.to(torch.complex64)
    elif is_synthetic_channel:
        if raw_h.ndim == 4 and raw_h.shape[1] == tcn_feat:
            half_c = raw_h.shape[1] // 2
            h_complex = torch.complex(raw_h[:, :half_c].float(), raw_h[:, half_c:].float())
        elif is_rayleigh:
            ch_shape = (B, tcn_feat // 2, h_feat, w_feat)
            hr = torch.randn(ch_shape, device=device, dtype=torch.float32)
            hi = torch.randn(ch_shape, device=device, dtype=torch.float32)
            h_complex = torch.complex(hr, hi) / np.sqrt(2.0)
        else:
            h_complex = torch.ones((B, tcn_feat // 2, h_feat, w_feat), device=device, dtype=torch.complex64)
    elif raw_h.ndim == 4:
        h_complex = torch.complex(raw_h[:, 0].float(), raw_h[:, 1].float())
    else:
        h_complex = raw_h.to(torch.complex64)

    # -------------------------------------------------------------
    # Step 1) ADJSCC Encoder Forward
    # -------------------------------------------------------------
    with torch.set_grad_enabled(update_encoder):
        if is_h_module:
            h_mean_real = torch.mean(h_complex.real.reshape(B, -1), dim=-1, keepdim=True)
            h_mean_imag = torch.mean(h_complex.imag.reshape(B, -1), dim=-1, keepdim=True)
            features = encoder(x_clean, snr_tensor, h_mean_real, h_mean_imag)
        else:
            features = encoder(x_clean, snr_tensor)

        # 복소수 변환 및 심볼 단위 전력 정규화 (논문 Eq. 2)
        f = features.reshape(B, -1)
        z_real = f[:, :dim_z]
        z_imag = f[:, dim_z:]
        z_in = torch.complex(z_real, z_imag)

        norm_factor = torch.sum(z_in.real ** 2 + z_in.imag ** 2, dim=1, keepdim=True)
        scale = torch.sqrt(dim_z / torch.clamp(norm_factor, min=1e-8))
        z_in = z_in * scale

        # 형태 매칭 (합성 채널: tcn_feat//2, h_feat, w_feat)
        if is_synthetic_channel:
            z_matched = z_in.reshape(B, tcn_feat // 2, h_feat, w_feat)
            if h_complex.shape != z_matched.shape:
                if is_rayleigh:
                    hr = torch.randn(z_matched.shape, device=device, dtype=torch.float32)
                    hi = torch.randn(z_matched.shape, device=device, dtype=torch.float32)
                    h_complex = torch.complex(hr, hi) / np.sqrt(2.0)
                else:
                    h_complex = torch.ones_like(z_matched)
        else:
            if dim_z == target_numel:
                z_matched = z_in.reshape(B, h_complex.shape[1], h_complex.shape[2])
            else:
                real_1d = z_in.real.unsqueeze(1)
                imag_1d = z_in.imag.unsqueeze(1)
                real_interp = F.interpolate(real_1d, size=target_numel, mode='linear', align_corners=False).squeeze(1)
                imag_interp = F.interpolate(imag_1d, size=target_numel, mode='linear', align_corners=False).squeeze(1)
                z_matched = torch.complex(real_interp, imag_interp).reshape(B, h_complex.shape[1], h_complex.shape[2])

        reduce_dims = tuple(range(1, z_matched.ndim))
        power_z = torch.mean(torch.abs(z_matched) ** 2, dim=reduce_dims, keepdim=True)
        z_matched = z_matched / torch.sqrt(torch.clamp(power_z, min=1e-12))

    # -------------------------------------------------------------
    # Step 2) 무선 채널 전송 및 노이즈 주입 (Y = H*Z + N)
    # -------------------------------------------------------------
    bcast_shape = [B] + [1] * (z_matched.ndim - 1)
    snr_bcast = snr_tensor.reshape(bcast_shape)
    ratio_snr = 10.0 ** (snr_bcast / 10.0)

    if is_equalized or is_awgn:
        clean_target_signal = z_matched
        sig_pow = torch.ones(bcast_shape, device=device, dtype=torch.float32)
    else:
        rx_noiseless = h_complex * z_matched
        clean_target_signal = rx_noiseless
        if is_rayleigh:
            sig_pow = torch.ones(bcast_shape, device=device, dtype=torch.float32)
        else:
            sig_pow = torch.mean(torch.abs(rx_noiseless) ** 2, dim=reduce_dims, keepdim=True)

    eff_sigma = torch.sqrt(sig_pow / (ratio_snr * 2.0)).float()
    noise_r = torch.randn(z_matched.shape, device=device, dtype=torch.float32) * eff_sigma
    noise_i = torch.randn(z_matched.shape, device=device, dtype=torch.float32) * eff_sigma
    rx_noise = torch.complex(noise_r, noise_i)

    eps = 1e-6
    h_mag = torch.abs(h_complex)
    h_safe = torch.where(h_mag < eps, eps * torch.exp(1j * torch.angle(h_complex)), h_complex)

    if is_equalized:
        eq_noise = rx_noise / h_safe
        rx_signal = clean_target_signal + eq_noise
        raw_sigma = (eff_sigma / torch.clamp(h_mag, min=eps)).float()
    else:
        rx_signal = clean_target_signal + rx_noise
        raw_sigma = eff_sigma.expand_as(z_matched.real).float()

    input_sigma = torch.mean(raw_sigma, dim=reduce_dims).float()

    if is_synthetic_channel:
        rx_input = torch.cat([rx_signal.real, rx_signal.imag], dim=1).float()
    else:
        rx_input = torch.stack([rx_signal.real, rx_signal.imag], dim=1).float()

    # Scale synthetic channel inputs to match diffusion model trained with data_norm (EDM sigma_data=0.5)
    if is_synthetic_channel and data_norm != 1.0:
        rx_input = rx_input / data_norm
        input_sigma = input_sigma / data_norm

    # -------------------------------------------------------------
    # Step 3) Channel-Adaptive Denoising Diffusion (CAD2M)
    # -------------------------------------------------------------
    orig_spatial_h, orig_spatial_w = rx_input.shape[-2], rx_input.shape[-1]
    if padding_power2:
        padded_signal, _, orig_shape = pad_collate_fn(rx_input)
        target_shape = (1,) + tuple(padded_signal.shape[1:])
        in_orig_shape = torch.tensor(tuple(orig_shape[0])).unsqueeze(0)
        padding_mask = padding_mask_from_original_shape(in_orig_shape, target_shape)
    else:
        padded_signal = rx_input
        padding_mask = 1.0

    # 디퓨전 업데이트 여부 및 역전파 흐름 제어
    grad_diffusion_enabled = update_diffusion or update_encoder
    with torch.set_grad_enabled(grad_diffusion_enabled):
        kw = sampler_kwargs or {}
        denoised_signal, _ = edm_sampler(
            net,
            latents=padded_signal,
            class_labels=snr_tensor,
            sigma_max=input_sigma,
            padding_mask=padding_mask,
            latents_already_noisy=True,
            num_steps=denoise_steps,
            **kw
        )

        if padding_power2:
            denoised_signal = denoised_signal[:, :, :orig_spatial_h, :orig_spatial_w]

        denoised_signal = denoised_signal.float()

        # De-normalize synthetic channel denoised output back to unit power for ADJSCC Decoder
        if is_synthetic_channel and data_norm != 1.0:
            denoised_signal = denoised_signal * data_norm

        # 디퓨전과 인코더가 모두 업데이트되지 않을 경우 gradient detach
        if not update_diffusion and not update_encoder:
            denoised_signal = denoised_signal.detach()

        if is_synthetic_channel:
            half_ch = denoised_signal.shape[1] // 2
            denoised_target_signal = torch.complex(denoised_signal[:, :half_ch], denoised_signal[:, half_ch:])
        else:
            denoised_target_signal = torch.complex(denoised_signal[:, 0], denoised_signal[:, 1])

        z_hat_complex = denoised_target_signal

        # 비등화 페이딩 채널의 경우 디퓨전 출력 신호에 채널 등화 수행 (논문 Eq. 38)
        if not is_equalized and not is_awgn:
            h_mag2 = torch.abs(h_complex) ** 2
            eps_eq = 1e-4
            z_hat_complex = (z_hat_complex * torch.conj(h_complex)) / (h_mag2 + eps_eq)

    # -------------------------------------------------------------
    # Step 4) ADJSCC Decoder Forward (이미지 복원)
    # -------------------------------------------------------------
    with torch.set_grad_enabled(update_decoder or update_diffusion or update_encoder):
        z_hat_complex = complex_nan_to_num(z_hat_complex, nan=0.0, posinf=10.0, neginf=-10.0)
        if is_synthetic_channel or target_numel == dim_z:
            z_unmatch = z_hat_complex.reshape(B, dim_z)
        else:
            r_flat = z_hat_complex.real.reshape(B, -1).unsqueeze(1)
            i_flat = z_hat_complex.imag.reshape(B, -1).unsqueeze(1)
            r_inv = F.interpolate(r_flat, size=dim_z, mode='linear', align_corners=False).squeeze(1)
            i_inv = F.interpolate(i_flat, size=dim_z, mode='linear', align_corners=False).squeeze(1)
            z_unmatch = torch.complex(r_inv, i_inv)

        if normalize_power:
            cur_pow = torch.mean(z_unmatch.real ** 2 + z_unmatch.imag ** 2, dim=1, keepdim=True)
            expected_pow = 1.0 + 10.0 ** (-snr_tensor / 10.0)
            z_unmatch = z_unmatch * torch.sqrt(expected_pow / torch.clamp(cur_pow, min=1e-8))

        f_dec = torch.cat([z_unmatch.real, z_unmatch.imag], dim=1).reshape(B, tcn_feat, h_feat, w_feat)
        f_dec = torch.clamp(f_dec, min=-10.0, max=10.0)

        if is_h_module:
            h_mean_real = torch.mean(h_complex.real.reshape(B, -1), dim=-1, keepdim=True)
            h_mean_imag = torch.mean(h_complex.imag.reshape(B, -1), dim=-1, keepdim=True)
            x_hat = decoder(f_dec, snr_tensor, h_mean_real, h_mean_imag)
        else:
            x_hat = decoder(f_dec, snr_tensor)

        x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=0.0)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

    return x_hat, x_clean, denoised_target_signal


# ============================================================================
# CLI Command Setup
# ============================================================================

@click.command()

# Core Stage 3 Options
@click.option('--epochs',            help='Number of training epochs (based on training images)', type=click.IntRange(min=1), default=20, show_default=True)
@click.option('--update_diffusion',   help='Whether to update diffusion model parameters', type=bool, default=True, show_default=True)
@click.option('--update_encoder',     help='Whether to update adaJSCC encoder parameters', type=bool, default=True, show_default=True)
@click.option('--update_decoder',     help='Whether to update adaJSCC decoder parameters', type=bool, default=True, show_default=True)
@click.option('--freeze_diffusion',   help='Convenience flag to freeze diffusion model (sets --update_diffusion=False)', is_flag=True)
@click.option('--freeze_encoder',     help='Convenience flag to freeze encoder (sets --update_encoder=False)', is_flag=True)
@click.option('--freeze_decoder',     help='Convenience flag to freeze decoder (sets --update_decoder=False)', is_flag=True)

# Checkpoints and Configurations
@click.option('--network_pkl',        help='Network pickle filename (.pkl) for pretrained SC diffusion model', metavar='PATH', type=str, required=True)
@click.option('--config_json',        help='Training options JSON filename (default: auto from network_pkl directory)', metavar='PATH', type=str, default=None)
@click.option('--adjscc_config',      help='Path to ADJSCC config JSON/YAML file (default: from config_json or fallback)', type=str, default=None)
@click.option('--adjscc_model_path',  help='Explicit path to ADJSCC checkpoint .pt file', type=str, default=None)

# Hyperparameters & Optimization
@click.option('--lr',                help='Default learning rate for active modules', type=float, default=1e-4, show_default=True)
@click.option('--lr_codec',          help='Explicit learning rate for Encoder and Decoder (overrides --lr)', type=float, default=None)
@click.option('--lr_diffusion',      help='Explicit learning rate for Diffusion model (overrides --lr)', type=float, default=None)
@click.option('--weight_decay',      help='Optimizer weight decay', type=float, default=0.0, show_default=True)
@click.option('--grad_clip',         help='Gradient norm clipping threshold', type=float, default=-1, show_default=True)
@click.option('--batch_size',        help='Total training batch size across all GPUs', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--batch_gpu',         help='Limit batch size per GPU (overrides --batch_size // world_size)', type=click.IntRange(min=1), default=None)
@click.option('--val_batch_size',    help='Batch size for validation', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--train_steps',       help='Diffusion sampling steps during training (default: 1 for fast CAD2M Stage 3)', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--val_steps',         help='Diffusion sampling steps during validation', type=click.IntRange(min=1), default=1, show_default=True)

# Channel & SNR Settings
@click.option('--channel_type',      help='Wireless channel type (awgn, rayleigh_fading, rayleigh_fading_eq, slow_fading, slow_fading_eq)', type=click.Choice(['slow_fading', 'slow_fading_eq', 'awgn', 'rayleigh_fading', 'rayleigh_fading_eq']), default=None)
@click.option('--slow_fading_eq',    help='Explicit flag for slow_fading_eq', type=bool, default=None)
@click.option('--snr_range',         help='SNR range (min,max) in dB for discrete integer uniform random sampling (e.g. "-2,10" or "0,20")', type=str, default=None)
@click.option('--fixed_codec_snr',   help='Fixed SNR (dB) conditioned to Encoder/Decoder during training/testing', type=float, default=None)
@click.option('--data',              help='Path to RENEW RF CSI dataset npz directory (required for slow_fading channels)', metavar='DIR', type=str, default=None)
@click.option('--data_norm',         help='Data normalization value for RF data', type=float, default=None)

# Dataset & Validation Options
@click.option('--cifar_data',        help='Root directory for CIFAR-10 dataset', type=str, default='./data', show_default=True)
@click.option('--image_data',        help='Path to custom image dataset directory (overrides CIFAR-10)', type=str, default=None)
@click.option('--dataset_keep_percentage', help='Percentage of dataset samples to use', type=float, default=1.0, show_default=True)
@click.option('--val_interval',      help='Validation frequency in epochs', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--val_rounds',        help='Number of evaluation rounds with different SNR and noise realizations (default: 4)', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--max_val_samples',   help='Maximum validation samples to evaluate', type=int, default=None)
@click.option('--skip_initial_val',  help='Skip initial pre-training validation (epoch 0)', is_flag=True)
@click.option('--no_normalize_power', help='Disable symbol power normalization before decoder (enabled by default)', is_flag=True)
@click.option('--padding_power2',    help='Pad channel dimensions to power of 2 for U-Net', is_flag=True)

# Sampling & Performance Options
@click.option('--rho',               help='EDM time step exponent rho', type=float, default=7.0, show_default=True)
@click.option('--S_churn',           help='EDM stochasticity strength S_churn', type=float, default=0.0, show_default=True)
@click.option('--S_min',             help='EDM stochasticity min noise level', type=float, default=0.0, show_default=True)
@click.option('--S_max',             help='EDM stochasticity max noise level', type=float, default=float('inf'), show_default=True)
@click.option('--S_noise',           help='EDM stochasticity noise inflation', type=float, default=1.0, show_default=True)
@click.option('--allow_tf32',         help='Enable TF32 matmul and cudnn', is_flag=True)
@click.option('--bf16',               help='Enable bfloat16 mixed precision', is_flag=True)
@click.option('--workers',           help='DataLoader worker threads', type=int, default=4, show_default=True)

# Output & Logging
@click.option('--outdir',            help='Directory to save fine-tuned checkpoints and options', metavar='DIR', type=str, default='training-runs/stage3', show_default=True)
@click.option('--expr_id',           help='Experiment name ID', type=str, default='stage3_finetune', show_default=True)
@click.option('--wandb', 'use_wandb', help='Enable Weights & Biases logging', type=bool, default=False, show_default=True)
@click.option('--wandb_group',       help='Weights & Biases group name', type=str, default='Stage3', show_default=True)
@click.option('--snap_epoch',        help='Save snapshot checkpoints every N epochs', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--seed',              help='Random seed for reproducibility', type=int, default=42, show_default=True)
def main(**kwargs):
    """SC-CAD2M Stage 3 Joint Fine-tuning."""
    # Line buffering for clean terminal logs
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    dist.init()
    device = torch.device('cuda')
    opts = dnnlib.EasyDict(kwargs)

    # Apply convenience freeze flags
    if opts.freeze_diffusion:
        opts.update_diffusion = False
    if opts.freeze_encoder:
        opts.update_encoder = False
    if opts.freeze_decoder:
        opts.update_decoder = False
    opts.normalize_power = not opts.no_normalize_power

    if opts.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    dist.print0('=' * 80)
    dist.print0('SC-CAD2M Stage 3 Joint Fine-tuning')
    dist.print0(f'Update Targets: Encoder={opts.update_encoder}, Diffusion={opts.update_diffusion}, Decoder={opts.update_decoder}')
    dist.print0(f'Epochs={opts.epochs}, Batch Size={opts.batch_size}, Train Steps K={opts.train_steps}')
    dist.print0(f'Validation: Interval={opts.val_interval} epoch(s), Rounds={opts.val_rounds}, Batch Size={opts.val_batch_size}')
    dist.print0(f'Normalize Power before decoder: {opts.normalize_power}')
    dist.print0('=' * 80)

    # -------------------------------------------------------------
    # 1. Load Training Options JSON from Diffusion Pretraining
    # -------------------------------------------------------------
    config_filepath = opts.config_json
    if config_filepath is None:
        config_filepath = os.path.join(os.path.dirname(opts.network_pkl), 'training_options.json')

    if os.path.isfile(config_filepath):
        dist.print0(f'Loading inherited training options from "{config_filepath}"...')
        with open(config_filepath, 'r', encoding='utf-8') as f:
            train_opts = json.load(f)
    else:
        dist.print0(f'Warning: training options file "{config_filepath}" not found. Using defaults.')
        train_opts = {}

    # Extract dataset options with CLI override
    ds_kwargs = train_opts.get('dataset_kwargs', {})
    inferred_channel_type = ds_kwargs.get('channel_type', 'slow_fading_eq')
    inferred_slow_fading_eq = ds_kwargs.get('slow_fading_eq', (inferred_channel_type in ('slow_fading_eq', 'rayleigh_fading_eq')))

    channel_type = str(opts.channel_type if opts.channel_type is not None else inferred_channel_type).lower().strip()
    slow_fading_eq = opts.slow_fading_eq if opts.slow_fading_eq is not None else (
        channel_type in ('slow_fading_eq', 'rayleigh_fading_eq') if opts.channel_type is not None else inferred_slow_fading_eq
    )
    if slow_fading_eq and opts.channel_type is None and channel_type == 'slow_fading':
        channel_type = 'slow_fading_eq'

    is_awgn = (channel_type == 'awgn')
    is_rayleigh = channel_type in ('rayleigh_fading', 'rayleigh_fading_eq')
    is_synthetic_channel = is_awgn or is_rayleigh
    is_equalized = bool(slow_fading_eq) or (channel_type in ('slow_fading_eq', 'rayleigh_fading_eq'))

    # RF Data path
    rf_data_path = opts.data if opts.data is not None else ds_kwargs.get('path', ds_kwargs.get('renew_path', None))
    if not is_synthetic_channel and rf_data_path is None:
        raise click.ClickException('--data is required unless channel_type is "awgn", "rayleigh_fading", or "rayleigh_fading_eq".')

    # Data normalization value
    if opts.data_norm is not None:
        data_norm = float(opts.data_norm)
    elif 'normalize_value' in ds_kwargs:
        data_norm = float(ds_kwargs['normalize_value'])
    elif is_synthetic_channel:
        data_norm = float(np.sqrt(2.0))
    else:
        data_norm = 1.0

    dist.print0(f'Channel Mode: type="{channel_type}", equalized={is_equalized}, synthetic={is_synthetic_channel}, data_norm={data_norm}')

    # SNR Range determination
    snr_range_str = opts.snr_range
    if snr_range_str is None:
        snr_range_str = ds_kwargs.get('encoder_snr_range', ds_kwargs.get('snr_range', '0,20'))
    snr_parts = [float(x.strip()) for x in snr_range_str.split(',')]
    snr_min, snr_max = snr_parts[0], snr_parts[1]
    dist.print0(f'Stage 3 Uniform SNR Sampling Range: [{snr_min:.1f}, {snr_max:.1f}] dB')

    # -------------------------------------------------------------
    # 2. Load Pretrained Diffusion Model
    # -------------------------------------------------------------
    dist.print0(f'Loading diffusion network from "{opts.network_pkl}"...')
    diff_data = save_utils.load_pkl(opts.network_pkl, verbose=(dist.get_rank() == 0))
    raw_net = diff_data.get('ema', diff_data.get('net'))
    if raw_net is None:
        raise ValueError(f"Pickle file '{opts.network_pkl}' does not contain 'ema' or 'net'.")

    net = copy.deepcopy(raw_net).to(device)
    ema = copy.deepcopy(diff_data.get('ema', net)).to(device).eval()
    for p in ema.parameters():
        p.requires_grad = False

    # Sync scheduler_mode with training_options.json (matching test_SC.py)
    target_sched_mode = train_opts.get('network_kwargs', {}).get(
        'scheduler_mode',
        train_opts.get('scheduler_mode', getattr(net, 'scheduler_mode', 'using_sigma_t_n'))
    )
    for m_obj in (net, ema):
        if hasattr(m_obj, 'scheduler_mode'):
            m_obj.scheduler_mode = target_sched_mode
        if hasattr(getattr(m_obj, 'noise_scheduler', None), 'scheduler_mode'):
            m_obj.noise_scheduler.scheduler_mode = target_sched_mode

    if opts.bf16 and hasattr(net, 'use_fp16'):
        net.use_fp16 = True

    # -------------------------------------------------------------
    # 3. Load ADJSCC Models (Encoder & Decoder)
    # -------------------------------------------------------------
    adjscc_cfg_path = opts.adjscc_config if opts.adjscc_config is not None else ds_kwargs.get('adjscc_config', 'misc/ADJSCC/config_example.json')
    adjscc_model_path = opts.adjscc_model_path if opts.adjscc_model_path is not None else ds_kwargs.get('adjscc_model_path', None)

    encoder, decoder, adjscc_cfg, tcn, is_h_module = load_adjscc_models(
        adjscc_config_path=adjscc_cfg_path,
        explicit_model_path=adjscc_model_path,
        device=device
    )

    T_steps = calculate_sc_t_from_config(adjscc_cfg, num_subcarriers=48)
    dist.print0(f'ADJSCC transmission time steps T = {T_steps} (tcn={tcn}, subcarriers=48)')

    # Compute feature map dimensions (assuming downsampling factor 4)
    h_in = adjscc_cfg.get('image_size', 32)
    if isinstance(h_in, (list, tuple)):
        h_in, w_in = h_in[-2], h_in[-1]
    else:
        h_in, w_in = int(h_in), int(h_in)
    h_feat = max(1, h_in // 4)
    w_feat = max(1, w_in // 4)
    tcn_feat = tcn
    dim_z = (tcn_feat * h_feat * w_feat) // 2
    target_numel = dim_z if is_synthetic_channel else (T_steps * 48)

    # -------------------------------------------------------------
    # 4. Setup Update / Freeze States
    # -------------------------------------------------------------
    if opts.update_encoder:
        encoder.train()
        for p in encoder.parameters():
            p.requires_grad = True
    else:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    if opts.update_diffusion:
        net.train()
        for p in net.parameters():
            p.requires_grad = True
    else:
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

    if opts.update_decoder:
        decoder.train()
        for p in decoder.parameters():
            p.requires_grad = True
    else:
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad = False

    # Multi-GPU DistributedDataParallel (DDP) setup
    if dist.get_world_size() > 1:
        dist.print0(f'Setting DistributedDataParallel across {dist.get_world_size()} GPUs...')
        if opts.update_encoder:
            encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[device], find_unused_parameters=True)
        if opts.update_diffusion:
            net = DDPNetWrapper(torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], find_unused_parameters=True))
        if opts.update_decoder:
            decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[device], find_unused_parameters=True)

    # -------------------------------------------------------------
    # 5. Optimizer Setup with Separated Parameter Groups
    # -------------------------------------------------------------
    lr_codec = opts.lr_codec if opts.lr_codec is not None else opts.lr
    lr_diff = opts.lr_diffusion if opts.lr_diffusion is not None else opts.lr

    param_groups = []
    if opts.update_encoder:
        param_groups.append({'params': encoder.parameters(), 'lr': lr_codec, 'name': 'encoder'})
        dist.print0(f'Optimizer: registered Encoder (lr={lr_codec})')
    if opts.update_diffusion:
        param_groups.append({'params': net.parameters(), 'lr': lr_diff, 'name': 'diffusion'})
        dist.print0(f'Optimizer: registered Diffusion U-Net (lr={lr_diff})')
    if opts.update_decoder:
        param_groups.append({'params': decoder.parameters(), 'lr': lr_codec, 'name': 'decoder'})
        dist.print0(f'Optimizer: registered Decoder (lr={lr_codec})')

    if not param_groups:
        raise click.ClickException('At least one of encoder, diffusion, or decoder must be updated!')

    optimizer = torch.optim.AdamW(param_groups, weight_decay=opts.weight_decay)

    # Batch size per GPU calculation
    batch_gpu = opts.batch_gpu
    if batch_gpu is None:
        batch_gpu = max(1, opts.batch_size // dist.get_world_size())
    val_batch_gpu = max(1, opts.val_batch_size // dist.get_world_size() if dist.get_world_size() > 1 else opts.val_batch_size)
    dist.print0(f'Batch size: total={opts.batch_size}, per_gpu={batch_gpu} across {dist.get_world_size()} GPUs')

    # -------------------------------------------------------------
    # 6. Load Datasets & DataLoaders
    # -------------------------------------------------------------
    dist.print0('Initializing Image datasets...')
    if opts.image_data is not None and os.path.exists(opts.image_data):
        from torchvision.datasets import ImageFolder
        transform = T.Compose([T.ToTensor()])
        train_img_ds = ImageFolder(os.path.join(opts.image_data, 'train'), transform=transform)
        val_img_ds = ImageFolder(os.path.join(opts.image_data, 'test'), transform=transform)
    else:
        train_img_ds = get_cifar10_dataset(opts.cifar_data, is_train=True)
        val_img_ds = get_cifar10_dataset(opts.cifar_data, is_train=False)

    renew_train_ch_ds = None
    renew_val_ch_ds = None
    if not is_synthetic_channel:
        dist.print0(f'Initializing RENEW RF Channel dataset from "{rf_data_path}"...')
        renew_train_ch_ds = RenewChannelDataset(
            renew_path=rf_data_path,
            adjscc_config=adjscc_cfg,
            T=T_steps,
            seed=opts.seed,
            dataset_keep_percentage=opts.dataset_keep_percentage,
            normalize_value=data_norm,
            channel_type=channel_type,
            slow_fading_eq=slow_fading_eq,
        )
        val_rf_data_path = train_opts.get('validation_kwargs', {}).get('validation_data', None)
        if val_rf_data_path is not None and os.path.exists(val_rf_data_path):
            renew_val_ch_ds = RenewChannelDataset(
                renew_path=val_rf_data_path,
                adjscc_config=adjscc_cfg,
                T=T_steps,
                seed=opts.seed + 1,
                dataset_keep_percentage=1.0,
                flip_keep_dataset=False,
                normalize_value=data_norm,
                channel_type=channel_type,
                slow_fading_eq=slow_fading_eq,
            )
        else:
            renew_val_ch_ds = RenewChannelDataset(
                renew_path=rf_data_path,
                adjscc_config=adjscc_cfg,
                T=T_steps,
                seed=opts.seed + 1,
                dataset_keep_percentage=opts.dataset_keep_percentage,
                flip_keep_dataset=(opts.dataset_keep_percentage < 1.0),
                normalize_value=data_norm,
                channel_type=channel_type,
                slow_fading_eq=slow_fading_eq,
            )

    paired_train_ds = PairedSCDataset(image_dataset=train_img_ds, renew_channel_dataset=renew_train_ch_ds)
    paired_val_ds = PairedSCDataset(image_dataset=val_img_ds, renew_channel_dataset=renew_val_ch_ds, max_samples=opts.max_val_samples)

    dist.print0(f'Dataset sizes: Train={len(paired_train_ds)}, Validation={len(paired_val_ds)}')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        paired_train_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, seed=opts.seed
    ) if dist.get_world_size() > 1 else None

    train_loader = DataLoader(
        dataset=paired_train_ds,
        sampler=train_sampler,
        batch_size=batch_gpu,
        shuffle=(train_sampler is None),
        num_workers=opts.workers,
        pin_memory=True,
        drop_last=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        paired_val_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    ) if dist.get_world_size() > 1 else None

    val_loader = DataLoader(
        dataset=paired_val_ds,
        sampler=val_sampler,
        batch_size=val_batch_gpu,
        shuffle=False,
        num_workers=opts.workers,
        pin_memory=True,
        drop_last=False
    )

    # -------------------------------------------------------------
    # 7. Metrics and Output Directory Setup
    # -------------------------------------------------------------
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=3, betas=(0.0448, 0.2856, 0.3001), normalize='relu'
    ).to(device)

    out_dir = os.path.join(opts.outdir, opts.expr_id)
    if dist.get_rank() == 0:
        os.makedirs(out_dir, exist_ok=True)
        # Inherit and save updated training_options.json to output directory
        merged_options = dict(train_opts)
        merged_options.update({
            'stage3_opts': dict(opts),
            'dataset_kwargs': dict(ds_kwargs, channel_type=channel_type, slow_fading_eq=slow_fading_eq),
            'adjscc_config': adjscc_cfg_path,
        })
        with open(os.path.join(out_dir, 'training_options.json'), 'w', encoding='utf-8') as f:
            json.dump(merged_options, f, indent=2)

    # Weights & Biases
    if dist.get_rank() == 0 and opts.use_wandb:
        try:
            import wandb
            wandb.init(project="ambient_rf_stage3", name=opts.expr_id, group=opts.wandb_group, config=opts)
            dist.print0('Weights & Biases initialized.')
        except Exception as e:
            dist.print0(f'Warning: Failed to initialize Wandb ({e}). Continuing without it.')
            opts.use_wandb = False
    sampler_kwargs = {
        'rho': opts.rho,
        'S_churn': opts.s_churn,
        'S_min': opts.s_min,
        'S_max': opts.s_max,
        'S_noise': opts.s_noise,
    }

    # -------------------------------------------------------------
    # 8. Validation Routine & Baseline Evaluation
    # -------------------------------------------------------------
    best_val_psnr = -1.0

    def run_validation(epoch_idx: int, train_metrics: Optional[Dict[str, Any]] = None) -> float:
        nonlocal best_val_psnr
        encoder.eval()
        net.eval()
        decoder.eval()
        psnr_metric.reset()
        msssim_metric.reset()

        eval_net = ema if opts.update_diffusion else net
        val_loss_sum = 0.0
        val_samples = 0

        # Preserve training RNG states and set deterministic base seed for repeatable validation across epochs
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device)
        base_val_seed = opts.seed + 10000 + dist.get_rank()

        val_rounds = max(1, int(opts.val_rounds))
        round_psnr_list = []

        with torch.inference_mode():
            for round_idx in range(val_rounds):
                # Set deterministic per-round seed: each round uses independent SNR and noise realizations,
                # but every epoch tests the exact same sequence of rounds for completely fair comparison
                round_seed = base_val_seed + round_idx * 1337
                torch.manual_seed(round_seed)
                torch.cuda.manual_seed(round_seed)

                round_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

                if val_rounds > 1:
                    round_desc = (
                        f'Epoch [{epoch_idx:03d}/{opts.epochs:03d}] Valid (Round {round_idx + 1}/{val_rounds})'
                        if epoch_idx > 0 else
                        f'Initial Baseline Valid [Epoch 000] (Round {round_idx + 1}/{val_rounds})'
                    )
                else:
                    round_desc = (
                        f'Epoch [{epoch_idx:03d}/{opts.epochs:03d}] Valid'
                        if epoch_idx > 0 else
                        'Initial Baseline Valid [Epoch 000]'
                    )

                val_pbar = tqdm.tqdm(
                    val_loader,
                    desc=round_desc,
                    disable=(dist.get_rank() != 0),
                    file=sys.stdout,
                    dynamic_ncols=True,
                    mininterval=0.5
                )

                for batch in val_pbar:
                    x_clean = batch['image'].to(device)
                    raw_h = batch['channel'].to(device)
                    B = x_clean.shape[0]

                    if opts.fixed_codec_snr is not None:
                        snr_tensor = torch.full((B, 1), float(opts.fixed_codec_snr), device=device, dtype=torch.float32)
                    else:
                        rand_snr = torch.randint(int(snr_min), int(snr_max) + 1, (B, 1), device=device, dtype=torch.float32)
                        snr_tensor = rand_snr

                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=opts.bf16):
                        x_hat, target_clean, _ = forward_stage3_pipeline(
                            x_clean=x_clean,
                            raw_h=raw_h,
                            snr_tensor=snr_tensor,
                            encoder=encoder,
                            net=eval_net,
                            decoder=decoder,
                            is_h_module=is_h_module,
                            is_synthetic_channel=is_synthetic_channel,
                            is_rayleigh=is_rayleigh,
                            is_equalized=is_equalized,
                            is_awgn=is_awgn,
                            target_numel=target_numel,
                            dim_z=dim_z,
                            tcn_feat=tcn_feat,
                            h_feat=h_feat,
                            w_feat=w_feat,
                            denoise_steps=opts.val_steps,
                            data_norm=data_norm,
                            padding_power2=opts.padding_power2,
                            normalize_power=opts.normalize_power,
                            update_encoder=False,
                            update_diffusion=False,
                            update_decoder=False,
                            sampler_kwargs=sampler_kwargs,
                        )

                        batch_loss = F.mse_loss(x_hat, target_clean).item()

                    val_loss_sum += batch_loss * B
                    val_samples += B
                    psnr_metric.update(x_hat, target_clean)
                    round_psnr_metric.update(x_hat, target_clean)
                    msssim_metric.update(x_hat, target_clean)

                round_psnr_val = round_psnr_metric.compute().item()
                round_psnr_list.append(round_psnr_val)

        # Restore training RNG states
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)

        val_psnr_val = psnr_metric.compute().item()
        val_msssim_val = msssim_metric.compute().item()

        if dist.get_world_size() > 1:
            loss_tensor = torch.tensor([val_loss_sum, float(val_samples)], device=device, dtype=torch.float64)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            val_mse_val = (loss_tensor[0] / torch.clamp(loss_tensor[1], min=1.0)).item()
        else:
            val_mse_val = val_loss_sum / max(val_samples, 1)

        prefix = f'Epoch [{epoch_idx:03d}/{opts.epochs:03d}] Valid Summary' if epoch_idx > 0 else 'Initial Baseline Valid Summary [Epoch 0]'
        rounds_str = f" (Rounds: [{', '.join(f'{p:.2f}' for p in round_psnr_list)}])" if val_rounds > 1 else ""
        dist.print0(
            f'{prefix}: '
            f'Mean PSNR: {val_psnr_val:.3f} dB{rounds_str} | MS-SSIM: {val_msssim_val:.4f} | MSE: {val_mse_val:.6f}'
        )

        # Wandb Logging
        if dist.get_rank() == 0 and opts.use_wandb:
            import wandb
            log_dict = {
                'epoch': epoch_idx,
                'val/mse': val_mse_val,
                'val/psnr': val_psnr_val,
                'val/msssim': val_msssim_val,
            }
            if train_metrics is not None:
                log_dict.update(train_metrics)
            wandb.log(log_dict)

        # Baseline saving (Epoch 0)
        if epoch_idx == 0:
            best_val_psnr = val_psnr_val
            if dist.get_rank() == 0:
                dist.print0(f'>>> Initial Baseline Validation PSNR established: {best_val_psnr:.3f} dB. Saving baseline checkpoints...')
                best_adjscc_path = os.path.join(out_dir, 'adjscc_best.pt')
                torch.save({'encoder': unwrap_module(encoder).state_dict(), 'decoder': unwrap_module(decoder).state_dict()}, best_adjscc_path)

                best_diff_path = os.path.join(out_dir, 'network-snapshot-best.pkl')
                diff_save_dict = {
                    'ema': copy.deepcopy(unwrap_module(ema)).cpu(),
                    'net': copy.deepcopy(unwrap_module(net)).cpu(),
                    'dataset_kwargs': dict(ds_kwargs),
                    'best_val_psnr': best_val_psnr,
                    'epoch': 0,
                }
                save_utils.save_pkl(diff_save_dict, best_diff_path)
                dist.print0(f'Saved baseline checkpoints: {best_adjscc_path} & {best_diff_path}')
        else:
            # During training epochs: check if improvement over best_val_psnr
            if val_psnr_val > best_val_psnr:
                if dist.get_rank() == 0:
                    dist.print0(
                        f'>>> New Best Validation PSNR: {val_psnr_val:.3f} dB '
                        f'(improved from {best_val_psnr:.3f} dB by +{val_psnr_val - best_val_psnr:.3f} dB)! Saving best checkpoints...'
                    )
                    best_adjscc_path = os.path.join(out_dir, 'adjscc_best.pt')
                    torch.save({'encoder': unwrap_module(encoder).state_dict(), 'decoder': unwrap_module(decoder).state_dict()}, best_adjscc_path)

                    best_diff_path = os.path.join(out_dir, 'network-snapshot-best.pkl')
                    diff_save_dict = {
                        'ema': copy.deepcopy(unwrap_module(ema)).cpu(),
                        'net': copy.deepcopy(unwrap_module(net)).cpu(),
                        'dataset_kwargs': dict(ds_kwargs),
                        'best_val_psnr': val_psnr_val,
                        'epoch': epoch_idx,
                    }
                    save_utils.save_pkl(diff_save_dict, best_diff_path)
                    dist.print0(f'Saved best checkpoints: {best_adjscc_path} & {best_diff_path}')
                best_val_psnr = val_psnr_val
            else:
                if dist.get_rank() == 0:
                    dist.print0(f'Current best validation PSNR remains: {best_val_psnr:.3f} dB (current: {val_psnr_val:.3f} dB)')

        if dist.get_world_size() > 1:
            torch.distributed.barrier()

        return val_psnr_val

    # Initial Baseline Validation before training starts
    if not opts.skip_initial_val:
        dist.print0('=' * 80)
        dist.print0('Running Initial Baseline Validation (Epoch 0) before training starts...')
        run_validation(0)
        dist.print0(f'Initial Baseline Established: PSNR = {best_val_psnr:.3f} dB')
        dist.print0('=' * 80)
    else:
        dist.print0('Skipping initial baseline validation (--skip_initial_val).')

    # -------------------------------------------------------------
    # 9. Main Training Loop
    # -------------------------------------------------------------
    start_time = time.time()

    for epoch in range(1, opts.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if opts.update_encoder:
            encoder.train()
        if opts.update_diffusion:
            net.train()
        if opts.update_decoder:
            decoder.train()

        running_loss = 0.0
        running_mse = 0.0
        train_samples = 0

        pbar = tqdm.tqdm(
            train_loader,
            desc=f'Epoch [{epoch:03d}/{opts.epochs:03d}] Train',
            disable=(dist.get_rank() != 0),
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.5
        )

        for batch in pbar:
            x_clean = batch['image'].to(device)
            raw_h = batch['channel'].to(device)
            B = x_clean.shape[0]

            # Uniform random discrete integer SNR sampling: zeta ~ U(snr_min, snr_max)
            if opts.fixed_codec_snr is not None:
                snr_tensor = torch.full((B, 1), float(opts.fixed_codec_snr), device=device, dtype=torch.float32)
            else:
                rand_snr = torch.randint(int(snr_min), int(snr_max) + 1, (B, 1), device=device, dtype=torch.float32)
                snr_tensor = rand_snr

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=opts.bf16):
                x_hat, target_clean, _ = forward_stage3_pipeline(
                    x_clean=x_clean,
                    raw_h=raw_h,
                    snr_tensor=snr_tensor,
                    encoder=encoder,
                    net=net,
                    decoder=decoder,
                    is_h_module=is_h_module,
                    is_synthetic_channel=is_synthetic_channel,
                    is_rayleigh=is_rayleigh,
                    is_equalized=is_equalized,
                    is_awgn=is_awgn,
                    target_numel=target_numel,
                    dim_z=dim_z,
                    tcn_feat=tcn_feat,
                    h_feat=h_feat,
                    w_feat=w_feat,
                    denoise_steps=opts.train_steps,
                    data_norm=data_norm,
                    padding_power2=opts.padding_power2,
                    normalize_power=opts.normalize_power,
                    update_encoder=opts.update_encoder,
                    update_diffusion=opts.update_diffusion,
                    update_decoder=opts.update_decoder,
                    sampler_kwargs=sampler_kwargs,
                )

                loss = F.mse_loss(x_hat, target_clean)

            loss.backward()

            if opts.grad_clip > 0:
                for grp in param_groups:
                    torch.nn.utils.clip_grad_norm_(grp['params'], max_norm=opts.grad_clip)

            optimizer.step()

            # Update Diffusion EMA if diffusion is trained
            if opts.update_diffusion:
                ema_beta = 0.999
                for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                    p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

            batch_mse = loss.item()
            running_loss += batch_mse * B
            running_mse += batch_mse * B
            train_samples += B

            current_mse = running_mse / max(train_samples, 1)
            current_psnr = 10.0 * np.log10(1.0 / max(current_mse, 1e-12))
            pbar.set_postfix({'MSE': f'{current_mse:.5f}', 'PSNR': f'{current_psnr:.2f}dB'})

        epoch_train_mse = running_mse / max(train_samples, 1)
        epoch_train_psnr = 10.0 * np.log10(1.0 / max(epoch_train_mse, 1e-12))
        dist.print0(f'Epoch [{epoch:03d}/{opts.epochs:03d}] Train Finished - MSE: {epoch_train_mse:.6f}, PSNR: {epoch_train_psnr:.2f} dB')

        # -------------------------------------------------------------
        # Validation Step
        # -------------------------------------------------------------
        if epoch % opts.val_interval == 0:
            run_validation(epoch, train_metrics={'train/mse': epoch_train_mse, 'train/psnr': epoch_train_psnr})
        else:
            if dist.get_rank() == 0 and opts.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/mse': epoch_train_mse,
                    'train/psnr': epoch_train_psnr,
                })

        # Snapshot saving
        if dist.get_rank() == 0 and (epoch % opts.snap_epoch == 0 or epoch == opts.epochs):
            snap_adjscc_path = os.path.join(out_dir, f'adjscc_epoch_{epoch:03d}.pt')
            torch.save({'encoder': unwrap_module(encoder).state_dict(), 'decoder': unwrap_module(decoder).state_dict()}, snap_adjscc_path)

            snap_diff_path = os.path.join(out_dir, f'network-snapshot-epoch_{epoch:03d}.pkl')
            diff_save_dict = {
                'ema': copy.deepcopy(unwrap_module(ema)).cpu(),
                'net': copy.deepcopy(unwrap_module(net)).cpu(),
                'dataset_kwargs': dict(ds_kwargs),
                'epoch': epoch,
            }
            save_utils.save_pkl(diff_save_dict, snap_diff_path)

            state_path = os.path.join(out_dir, f'training-state-epoch_{epoch:03d}.pt')
            save_utils.save_pt({
                'epoch': epoch,
                'optimizer_state': optimizer.state_dict(),
                'best_val_psnr': best_val_psnr,
            }, state_path)

    total_time = time.time() - start_time
    dist.print0('=' * 80)
    dist.print0(f'Stage 3 Fine-Tuning Completed in {total_time / 60.0:.2f} minutes!')
    dist.print0(f'Best Validation PSNR: {best_val_psnr:.3f} dB')
    dist.print0(f'Final Checkpoints available in: {out_dir}')
    dist.print0('=' * 80)


if __name__ == '__main__':
    main()

