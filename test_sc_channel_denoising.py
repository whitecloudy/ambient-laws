#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sc_channel_denoising.py - Semantic Communications (SC) Evaluation with Diffusion Channel Denoising

Pipeline:
1) Sample 14 consecutive time slots from the RENEW channel dataset to form h_true of shape (B, 16, 14, 52)
   [16 channels = 8 Antennas * 2 (Real/Imag), 14 Time slots, 52 Subcarriers].
2) Compute per-sample reference channel power P_h = E[|h_true_sub|^2] over the SC-mapped sub-channel h_true_sub,
   and add complex AWGN with standard deviation sigma_n = sqrt(P_h / 10^(SNR_dB / 10)) (real/imag std = sigma_n / sqrt(2))
   to h_true to obtain h_noisy, matching the wireless transmission AWGN variance under E[|z|^2] = 1.
3) Denoise h_noisy using the pretrained RENEW Diffusion EDM sampler to obtain reconstructed channel h_hat (B, 16, 14, 52).
4) Convert h_true, h_hat (and h_noisy) to complex tensors of shape (B, 8, 14, 52), select 1 random antenna out of 8
   and 48 valid data subcarriers out of 52.
5) Verify SC encoder complex symbol count num_symbols:
   T = ceil(num_symbols / 48). If T > 14 (num_symbols > 672), raise a ValueError.
   Otherwise, slice T consecutive slots starting at random t_start (0 <= t_start <= 14 - T) and map the first
   num_symbols to 1D complex channel vectors h_true_sub, h_hat_sub.
6) Normalize SC encoder latent z so E[|z|^2] = 1, transmit over wireless channel y = z * h_true_sub + n,
   perform Zero-Forcing (ZF) equalization z_hat = y / (h_hat_sub + 1e-8), and decode via SC decoder.
"""

import os
import sys
import csv
import json
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import click
import numpy as np
import torch
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
from training.dataset import renewRfProcessedDataset
from training.semantic_comm_dataset import (
    _load_adjscc_config,
    find_adjscc_checkpoint,
    DATA_SUBCARRIER_INDICES_48,
)

# Reuse helper functions from test_SC.py and denoising_tester.py
from test_SC import (
    _power2ceil,
    dB_to_ratio,
    ratio_to_dB,
    cal_SNR,
    pad_collate_fn,
    load_adjscc_models,
    get_cifar10_test_dataset,
)

DIFFUSION_TIME_SLOTS = 14
DIFFUSION_SUBCARRIERS = 52
VALID_SUBCARRIERS = 48
DIFFUSION_ANTENNAS = 8
MAX_DIFFUSION_SYMBOLS = DIFFUSION_TIME_SLOTS * VALID_SUBCARRIERS  # 14 * 48 = 672


def verify_symbol_capacity(num_symbols: int) -> int:
    """
    Verify that the required time slots T = ceil(num_symbols / 48) do not exceed
    the diffusion model capacity (14 slots, 672 symbols).
    """
    T = int(math.ceil(num_symbols / float(VALID_SUBCARRIERS)))
    if T > DIFFUSION_TIME_SLOTS or num_symbols > MAX_DIFFUSION_SYMBOLS:
        raise ValueError(
            f"Error: Required time slots (T={T}) exceeds the diffusion model capacity (14 slots, 672 symbols)."
        )
    return max(1, T)


class PairedSCChannelDenoisingDataset(Dataset):
    """
    Pairs CIFAR-10 test images with 14-time-slot RENEW channel tensors of shape (16, 14, 52).
    For each sample idx, deterministically assigns (via seed + idx):
      - RENEW file index & 14-slot starting frame (if the file has > 14 frames)
      - Random antenna index in [0, 7]
      - Random time slot offset t_start in [0, 14 - T]
      - Per-sample noise seed so all SNR steps evaluate on identical pairings
    """

    def __init__(
        self,
        image_dataset: Dataset,
        renew_dataset: renewRfProcessedDataset,
        required_T: int,
        seed: int = 11454,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.image_dataset = image_dataset
        self.renew_dataset = renew_dataset
        self.required_T = int(required_T)
        self.seed = int(seed)

        self.n_images = len(image_dataset)
        self.n_channels = len(renew_dataset)
        if self.n_channels == 0:
            raise ValueError("RENEW dataset is empty!")

        total = self.n_images
        if max_samples is not None:
            total = min(total, int(max_samples))
        self.length = total

    def __len__(self) -> int:
        return self.length

    def _extract_14_slots(self, raw_h: torch.Tensor, rng: np.random.RandomState) -> torch.Tensor:
        """
        Ensure raw_h has shape (16, 14, 52) by slicing 14 consecutive time slots
        from a random starting time slot if the time dimension exceeds 14.
        """
        if raw_h.ndim != 3:
            raise ValueError(f"Expected 3D channel tensor (16, T_frames, 52), got shape {tuple(raw_h.shape)}")

        c_dim, t_frames, sub_dim = raw_h.shape
        if c_dim != DIFFUSION_ANTENNAS * 2 or sub_dim != DIFFUSION_SUBCARRIERS:
            raise ValueError(
                f"Expected channel tensor shape (16, >=14, 52), got {tuple(raw_h.shape)}. "
                f"Check that transpose=[1, 0], view_as_complex=False, complex_merge_axis=0."
            )

        if t_frames == DIFFUSION_TIME_SLOTS:
            return raw_h
        elif t_frames > DIFFUSION_TIME_SLOTS:
            f_start = int(rng.randint(0, t_frames - DIFFUSION_TIME_SLOTS + 1))
            return raw_h[:, f_start : f_start + DIFFUSION_TIME_SLOTS, :]
        else:
            # Pad along time axis if fewer than 14 frames exist in the file
            pad_t = DIFFUSION_TIME_SLOTS - t_frames
            return F.pad(raw_h, (0, 0, 0, pad_t), mode='replicate')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_img = self.image_dataset[idx]
        if isinstance(item_img, (tuple, list)):
            img = item_img[0]
        elif isinstance(item_img, dict):
            img = item_img.get('image', item_img.get('img'))
        else:
            img = item_img

        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)

        # Deterministic per-sample RNG
        rng = np.random.RandomState((self.seed + int(idx) * 10007) % (2**31 - 1))

        ch_idx = idx % self.n_channels
        ch_item = self.renew_dataset[ch_idx]
        raw_h = ch_item['image']
        if not isinstance(raw_h, torch.Tensor):
            raw_h = torch.from_numpy(raw_h)
        raw_h = raw_h.float()

        h_true_14 = self._extract_14_slots(raw_h, rng)  # (16, 14, 52)

        antenna_idx = int(rng.randint(0, DIFFUSION_ANTENNAS))
        max_t_start = DIFFUSION_TIME_SLOTS - self.required_T
        t_start = int(rng.randint(0, max_t_start + 1)) if max_t_start > 0 else 0
        sample_noise_seed = int(rng.randint(0, 2**31 - 1))

        return {
            'image': img.float(),
            'h_true': h_true_14,
            'antenna_idx': antenna_idx,
            't_start': t_start,
            'noise_seed': sample_noise_seed,
            'idx': idx,
        }


def generate_batch_gaussian_noise(
    shape: Tuple[int, ...],
    noise_seeds: torch.Tensor,
    seed_offset: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic standard normal noise per sample in the batch."""
    B = shape[0]
    sample_shape = shape[1:]
    noises = []
    for b in range(B):
        gen = torch.Generator(device='cpu').manual_seed((int(noise_seeds[b].item()) + seed_offset) % (2**31 - 1))
        noises.append(torch.randn(sample_shape, generator=gen, dtype=dtype))
    return torch.stack(noises, dim=0).to(device=device, dtype=dtype)


def real16_to_complex8(h_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert (B, 16, 14, 52) real/imag merged tensor (channels 0..7 real, 8..15 imag)
    into complex64 tensor of shape (B, 8, 14, 52).
    """
    if h_tensor.ndim != 4 or h_tensor.shape[1] != DIFFUSION_ANTENNAS * 2:
        raise ValueError(f"Expected shape (B, 16, 14, 52), got {tuple(h_tensor.shape)}")
    h_real, h_imag = torch.chunk(h_tensor.float(), 2, dim=1)
    return torch.complex(h_real, h_imag)


def slice_channel_resources(
    h_complex: torch.Tensor,
    antenna_indices: torch.Tensor,
    t_starts: torch.Tensor,
    T: int,
    num_symbols: int,
    subcarrier_indices: torch.Tensor,
) -> torch.Tensor:
    """
    From complex channel tensor (B, 8, 14, 52):
      1) Select 1 antenna per sample -> (B, 14, 52)
      2) Select 48 valid subcarriers -> (B, 14, 48)
      3) Slice T consecutive time slots starting at t_start[b] -> (B, T, 48)
      4) Flatten to (B, T * 48) and slice the first num_symbols -> (B, num_symbols)
    """
    B = h_complex.shape[0]
    batch_arange = torch.arange(B, device=h_complex.device)

    # 1) Select 1 antenna per sample: (B, 14, 52)
    h_ant = h_complex[batch_arange, antenna_indices]

    # 2) Select 48 valid subcarriers: (B, 14, 48)
    h_sub48 = torch.index_select(h_ant, dim=-1, index=subcarrier_indices)

    # 3) Slice T time slots [t_start : t_start + T] per sample: (B, T, 48)
    t_offsets = torch.arange(T, device=h_complex.device).unsqueeze(0)  # (1, T)
    time_idx = t_starts.unsqueeze(1) + t_offsets  # (B, T)
    time_idx_expanded = time_idx.unsqueeze(-1).expand(B, T, VALID_SUBCARRIERS)
    h_slots = torch.gather(h_sub48, dim=1, index=time_idx_expanded)  # (B, T, 48)

    # 4) Map to 1D of length num_symbols: (B, num_symbols)
    return h_slots.reshape(B, T * VALID_SUBCARRIERS)[:, :num_symbols]


# ----------------------------------------------------------------------------
# CLI Command Setup
# ----------------------------------------------------------------------------

@click.command()
@click.option('--network_pkl',              help='RENEW channel diffusion network pickle filename (.pkl)', metavar='PATH', type=str, required=True)
@click.option('--config_json',              help='Diffusion training options JSON filename', metavar='PATH', type=str, default=None)
@click.option('--adjscc_config',            help='Path to ADJSCC config JSON/YAML file', type=str, default='misc/ADJSCC/config_example.json', show_default=True)
@click.option('--adjscc_model_path',        help='Explicit path to ADJSCC checkpoint .pt', type=str, default=None)
@click.option('--data',                     help='Path to RENEW RF CSI dataset npz directory', metavar='DIR', type=str, required=True)
@click.option('--data_norm',                help='Data normalization value for RENEW RF data (default: auto from training_options.json)', type=float, default=None)
@click.option('--cifar_data',               help='Root path for CIFAR-10 test data', type=str, default='./data', show_default=True)
@click.option('--using_train_image',        help='Use CIFAR-10 train set instead of test set', is_flag=True)
@click.option('--data_keep_ratio',          help='Data keep ratio for RENEW channel dataset', type=float, default=1.0, show_default=True)
@click.option('--must_contain',             help='Filter RENEW filenames to contain substring/pattern', type=str, default=None)
@click.option('--must_not_contain',         help='Filter RENEW filenames not to contain substring/pattern', type=str, default=None)
@click.option('--flip_dataset',             help='Use held-out/removed split of RENEW data', is_flag=True)
@click.option('--test_SNR_range', 'test_SNR_range', help='SNR range to test in dB (e.g. "0,20")', type=str, default='0,20', show_default=True)
@click.option('--test_SNR_step', 'test_SNR_step',   help='SNR step size (dB)', type=float, default=2.0, show_default=True)
@click.option('--fixed_codec_snr',          help='Fixed SNR (dB) conditioned to ADJSCC Encoder/Decoder', type=float, default=None)
@click.option('--batch_size',               help='Batch size for evaluation', type=int, default=64, show_default=True)
@click.option('--max_samples',              help='Maximum test samples (default: full CIFAR-10 test set)', type=int, default=None)
@click.option('--steps', 'num_steps',      help='Number of EDM sampling steps', type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level', type=float, default=None)
@click.option('--sigma_max',               help='Highest noise level override', type=float, default=None)
@click.option('--rho',                     help='Time step exponent', type=float, default=7.0, show_default=True)
@click.option('--S_churn', 'S_churn',     help='Stochasticity strength', type=float, default=0.0, show_default=True)
@click.option('--S_min', 'S_min',         help='Stochasticity min noise level', type=float, default=0.0, show_default=True)
@click.option('--S_max', 'S_max',         help='Stochasticity max noise level', type=float, default=float('inf'), show_default=True)
@click.option('--S_noise', 'S_noise',     help='Stochasticity noise inflation', type=float, default=1.0, show_default=True)
@click.option('--bf16',                    help='Enable bfloat16 precision for diffusion sampling', is_flag=True)
@click.option('--allow_tf32',              help='Allow TF32 matmul and cudnn', is_flag=True)
@click.option('--no_baselines',            help='Disable baseline evaluations (NoDiff, PerfectCSI, Clean)', is_flag=True)
@click.option('--save_images',             help='Save visual comparison image grids (.png) for each SNR', is_flag=True)
@click.option('--output',                   help='Output CSV filename or directory', type=str, default=None)
@click.option('--seed',                     help='Random seed for reproducibility', type=int, default=11454, show_default=True)
@click.option('--force_no_nn_scheduler',   help='Force noise scheduling to operate in no_nn_scheduler mode', is_flag=True)
@click.option('--scheduler_mode',           help='Override scheduler mode for the network', type=str, default=None)
@click.option('--padding_power2',          help='Pad spatial dimensions to power of 2 for U-Net during diffusion sampling', is_flag=True)
@click.option('--normalize_power',         help='Enable symbol power re-normalization before ADJSCC decoder (disabled by default)', is_flag=True)
def main(**kwargs):
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    dist.init()
    device = torch.device('cuda')
    opt = dnnlib.EasyDict(kwargs)

    if dist.get_rank() != 0:
        torch.distributed.barrier()

    torch.backends.cudnn.allow_tf32 = bool(opt.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(opt.allow_tf32)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # 1. Load ADJSCC Models & Verify Symbol Capacity Early
    encoder, decoder, adjscc_cfg, tcn, is_h_module = load_adjscc_models(
        adjscc_config_path=opt.adjscc_config,
        explicit_model_path=opt.adjscc_model_path,
        device=device,
    )

    # Probe encoder output symbol count with a dummy image to verify T <= 14 upfront
    img_size_cfg = adjscc_cfg.get('image_size', adjscc_cfg.get('img_size', [32, 32]))
    if isinstance(img_size_cfg, (int, float)):
        img_h, img_w = int(img_size_cfg), int(img_size_cfg)
    else:
        img_h, img_w = int(img_size_cfg[-2]), int(img_size_cfg[-1])
    in_channels = int(adjscc_cfg.get('in_channels', 3))

    with torch.inference_mode():
        dummy_x = torch.zeros((1, in_channels, img_h, img_w), device=device, dtype=torch.float32)
        dummy_snr = torch.zeros((1, 1), device=device, dtype=torch.float32)
        if is_h_module:
            dummy_h = torch.ones((1, 1), device=device, dtype=torch.float32)
            dummy_feat = encoder(dummy_x, dummy_snr, dummy_h, torch.zeros_like(dummy_h))
        else:
            dummy_feat = encoder(dummy_x, dummy_snr)
        expected_num_symbols = dummy_feat.numel() // 2

    required_T = verify_symbol_capacity(expected_num_symbols)
    dist.print0(
        f'ADJSCC Complex Symbols: num_symbols = {expected_num_symbols} -> '
        f'Required time slots T = {required_T} <= {DIFFUSION_TIME_SLOTS} (max {MAX_DIFFUSION_SYMBOLS} symbols)'
    )

    # 2. Load RENEW Channel Diffusion Model
    dist.print0(f'Loading RENEW channel diffusion network from "{opt.network_pkl}"...')
    pkl_data = save_utils.load_pkl(opt.network_pkl, verbose=(dist.get_rank() == 0))
    net = pkl_data['ema'].to(device)
    if opt.bf16:
        if hasattr(net, 'use_fp16'):
            net.use_fp16 = True
        if hasattr(getattr(net, 'module', None), 'use_fp16'):
            net.module.use_fp16 = True

    if opt.config_json is not None:
        config_filepath = opt.config_json
    else:
        config_filepath = os.path.join(os.path.dirname(opt.network_pkl), 'training_options.json')

    if os.path.isfile(config_filepath):
        dist.print0(f'Loading diffusion training options from "{config_filepath}"...')
        with open(config_filepath, 'r', encoding='utf-8') as f:
            train_opts = json.load(f)
    else:
        dist.print0('Warning: training_options.json not found. Using default RENEW dataset arguments.')
        train_opts = {}

    # Restore / configure noise scheduler mode (matching denoising_tester.py)
    target_mode = (
        opt.scheduler_mode
        if opt.scheduler_mode is not None
        else train_opts.get('network_kwargs', {}).get(
            'scheduler_mode',
            train_opts.get('scheduler_mode', getattr(net, 'scheduler_mode', 'using_sigma_t_n')),
        )
    )
    dist.print0(f'Setting scheduler_mode to "{target_mode}"...')
    net.scheduler_mode = target_mode
    if hasattr(net, 'noise_scheduler'):
        net.noise_scheduler.scheduler_mode = target_mode
        if hasattr(net.noise_scheduler, 'noise_decoder'):
            net.noise_scheduler.noise_decoder.scheduler_mode = target_mode
    elif hasattr(getattr(net, 'module', None), 'noise_scheduler'):
        net.module.noise_scheduler.scheduler_mode = target_mode
        if hasattr(net.module.noise_scheduler, 'noise_decoder'):
            net.module.noise_scheduler.noise_decoder.scheduler_mode = target_mode

    if opt.force_no_nn_scheduler:
        dist.print0('Forcing noise scheduler to "no_nn_scheduler" mode.')
        net.scheduler_mode = 'no_nn_scheduler'
        if hasattr(net, 'noise_scheduler'):
            net.noise_scheduler.scheduler_mode = 'no_nn_scheduler'
            if hasattr(net.noise_scheduler, 'noise_decoder'):
                net.noise_scheduler.noise_decoder.scheduler_mode = 'no_nn_scheduler'
        elif hasattr(getattr(net, 'module', None), 'noise_scheduler'):
            net.module.noise_scheduler.scheduler_mode = 'no_nn_scheduler'
            if hasattr(net.module.noise_scheduler, 'noise_decoder'):
                net.module.noise_scheduler.noise_decoder.scheduler_mode = 'no_nn_scheduler'

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # 3. Initialize RENEW Dataset (16, 14, 52) & CIFAR-10 Dataset
    saved_ds_kwargs = dict(train_opts.get('dataset_kwargs', {}))
    normalize_value = (
        float(opt.data_norm)
        if opt.data_norm is not None
        else float(saved_ds_kwargs.get('normalize_value', 1.0))
    )
    dist.print0(f'Initializing RENEW RF dataset from "{opt.data}" (normalize_value={normalize_value})...')

    renew_ds_kwargs = dnnlib.EasyDict(
        path=opt.data,
        use_labels=False,
        cache=False,
        sigma=0.0,
        corruption_probability_per_image=0.0,
        corruption_probability_per_pixel=1.0,
        only_positive=False,
        view_as_complex=False,
        complex_merge_axis=0,
        resolution=(DIFFUSION_TIME_SLOTS, DIFFUSION_ANTENNAS),
        transpose=[1, 0],
        normalize_value=normalize_value,
        must_contain=opt.must_contain,
        must_not_contain=opt.must_not_contain,
        dataset_keep_percentage=opt.data_keep_ratio,
        flip_keep_dataset=bool(opt.flip_dataset),
        multiply_noise_sigma=1.0,
        additive_noise_sigma=0.0,
        only_additive_noise=False,
        noise_mean_flag=False,
        snr_range=None,
    )
    renew_ds = renewRfProcessedDataset(**renew_ds_kwargs)

    dist.print0('Initializing CIFAR-10 dataset...')
    cifar_test_ds = get_cifar10_test_dataset(opt.cifar_data, using_train_image=opt.using_train_image)

    paired_test_ds = PairedSCChannelDenoisingDataset(
        image_dataset=cifar_test_ds,
        renew_dataset=renew_ds,
        required_T=required_T,
        seed=opt.seed,
        max_samples=opt.max_samples,
    )
    dist.print0(f'Total paired test samples to evaluate: {len(paired_test_ds)}')
    dist.print0(f'Normalize symbol power before decoder: {opt.normalize_power}')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        paired_test_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )
    dataloader = DataLoader(
        dataset=paired_test_ds,
        sampler=dist_sampler,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Setup Evaluation Metrics (torchmetrics)
    psnr_diff_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    msssim_diff_metric = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=3, betas=(0.0448, 0.2856, 0.3001), normalize='relu'
    ).to(device)

    eval_baselines = not opt.no_baselines
    if eval_baselines:
        psnr_nodiff_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        msssim_nodiff_metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=3, betas=(0.0448, 0.2856, 0.3001), normalize='relu'
        ).to(device)
        psnr_trueh_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        msssim_trueh_metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=3, betas=(0.0448, 0.2856, 0.3001), normalize='relu'
        ).to(device)
        psnr_clean_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        msssim_clean_metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=3, betas=(0.0448, 0.2856, 0.3001), normalize='relu'
        ).to(device)

    # 5. Parse SNR Steps & Sampler Kwargs
    snr_range = [float(x) for x in opt.test_SNR_range.split(',')]
    snr_steps = np.arange(snr_range[0], snr_range[1] + 1e-5, opt.test_SNR_step)
    dist.print0(f'Target SNR range: {snr_steps.tolist()} dB')

    sampler_kwargs = {
        'num_steps': opt.num_steps,
        'sigma_min': opt.sigma_min,
        'rho': opt.rho,
        'S_churn': opt.S_churn,
        'S_min': opt.S_min,
        'S_max': opt.S_max,
        'S_noise': opt.S_noise,
    }
    sampler_kwargs = {k: v for k, v in sampler_kwargs.items() if v is not None}

    subcarrier_idx_tensor = torch.tensor(DATA_SUBCARRIER_INDICES_48, dtype=torch.long, device=device)

    # Setup Output Paths
    if opt.output is not None:
        out_csv = opt.output if opt.output.endswith('.csv') else os.path.join(opt.output, 'eval_sc_channel_denoising.csv')
    else:
        out_dir = os.path.dirname(opt.network_pkl)
        out_csv = os.path.join(out_dir, 'eval_sc_channel_denoising.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out_json = out_csv.replace('.csv', '.json')

    results_table: List[Dict[str, Any]] = []
    headers = [
        'SNR_dB',
        'Ch_SNR_Diff',
        'Sig_SNR_Diff',
        'PSNR_Diff',
        'MSSSIM_Diff',
    ]
    if eval_baselines:
        headers.extend([
            'Ch_SNR_NoDiff',
            'Sig_SNR_NoDiff',
            'PSNR_NoDiff',
            'MSSSIM_NoDiff',
            'Sig_SNR_PerfectCSI',
            'PSNR_PerfectCSI',
            'MSSSIM_PerfectCSI',
            'PSNR_Clean',
            'MSSSIM_Clean',
        ])

    # Print Table Header
    dist.print0('\n' + '=' * 155)
    if eval_baselines:
        dist.print0(
            f'{"SNR(dB)":>7} | '
            f'{"ChSNR(Diff)":>11} | {"SigSNR(Diff)":>12} | {"PSNR(Diff)":>10} | {"MSSSIM(Diff)":>12} | '
            f'{"ChSNR(NoDf)":>11} | {"PSNR(NoDf)":>10} | {"MSSSIM(NoDf)":>12} | '
            f'{"PSNR(TrueH)":>11} | {"MSSSIM(TrueH)":>13} | '
            f'{"PSNR(Clean)":>11} | {"MSSSIM(Clean)":>13}'
        )
        dist.print0('-' * 155)
    else:
        dist.print0(
            f'{"SNR(dB)":>7} | '
            f'{"ChSNR(Diff)":>11} | {"SigSNR(Diff)":>12} | {"PSNR(Diff)":>10} | {"MSSSIM(Diff)":>12}'
        )
        dist.print0('-' * 65)

    # 6. Main Evaluation Loop over SNR Steps
    with torch.inference_mode(True):
        for snr_db in snr_steps:
            snr_db_val = float(snr_db)
            codec_snr_val = float(opt.fixed_codec_snr) if opt.fixed_codec_snr is not None else snr_db_val

            psnr_diff_metric.reset()
            msssim_diff_metric.reset()
            if eval_baselines:
                psnr_nodiff_metric.reset()
                msssim_nodiff_metric.reset()
                psnr_trueh_metric.reset()
                msssim_trueh_metric.reset()
                psnr_clean_metric.reset()
                msssim_clean_metric.reset()

            ch_snr_diff_ratio_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            sig_snr_diff_ratio_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            ch_snr_nodiff_ratio_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            sig_snr_nodiff_ratio_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            sig_snr_trueh_ratio_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            total_eval_samples = torch.tensor(0, device=device, dtype=torch.long)

            saved_grid_images = None

            # Target linear SNR ratio: 10^(SNR_dB / 10)
            ratio_snr = dB_to_ratio(snr_db_val)

            tqdm_bar = tqdm.tqdm(
                dataloader,
                desc=f'Evaluating SNR {snr_db_val:4.1f} dB',
                unit='batch',
                disable=(dist.get_rank() != 0),
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.2,
            )

            for batch in tqdm_bar:
                x_clean = batch['image'].to(device)               # (B, 3, 32, 32)
                h_true = batch['h_true'].to(device).float()       # (B, 16, 14, 52)
                antenna_idx = batch['antenna_idx'].to(device, dtype=torch.long)  # (B,)
                t_start = batch['t_start'].to(device, dtype=torch.long)          # (B,)
                noise_seeds = batch['noise_seed']                 # (B,)
                B = x_clean.shape[0]

                # ------------------------------------------------------------------
                # Step 1: Extract h_true_sub & Compute Channel-Power-Scaled sigma_n
                # ------------------------------------------------------------------
                h_true_complex = real16_to_complex8(h_true)       # (B, 8, 14, 52)
                h_true_sub = slice_channel_resources(
                    h_true_complex, antenna_idx, t_start, required_T, expected_num_symbols, subcarrier_idx_tensor
                )  # (B, expected_num_symbols)

                # Per-sample reference channel power P_h = E[|h_true_sub|^2]
                sig_pow_h = torch.mean(torch.abs(h_true_sub) ** 2, dim=1)  # (B,)
                # Complex AWGN std sigma_n = sqrt(P_h / ratio_snr), per real/imag std = sigma_n / sqrt(2)
                sigma_per_real_imag = torch.sqrt(sig_pow_h / (ratio_snr * 2.0)).float()  # (B,)

                ch_unit_noise = generate_batch_gaussian_noise(
                    shape=tuple(h_true.shape),
                    noise_seeds=noise_seeds,
                    seed_offset=1001,
                    device=device,
                    dtype=torch.float32,
                )
                h_noisy = h_true + ch_unit_noise * sigma_per_real_imag.view(B, 1, 1, 1)  # (B, 16, 14, 52)

                if opt.sigma_max is not None:
                    input_sigma = torch.full((B,), float(opt.sigma_max), device=device, dtype=torch.float32)
                else:
                    input_sigma = sigma_per_real_imag

                orig_h, orig_w = h_noisy.shape[-2], h_noisy.shape[-1]
                if opt.padding_power2:
                    padded_h_noisy, _, orig_shape = pad_collate_fn(h_noisy)
                    target_shape = (1,) + tuple(padded_h_noisy.shape[1:])
                    in_orig_shape = torch.tensor(tuple(orig_shape[0])).unsqueeze(0)
                    padding_mask = padding_mask_from_original_shape(in_orig_shape, target_shape)
                else:
                    padded_h_noisy = h_noisy
                    padding_mask = 1

                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=opt.bf16):
                    h_hat_padded, _ = edm_sampler(
                        net,
                        latents=padded_h_noisy,
                        sigma_max=input_sigma,
                        padding_mask=padding_mask,
                        latents_already_noisy=True,
                        **sampler_kwargs,
                    )

                if opt.padding_power2:
                    h_hat = h_hat_padded[:, :, :orig_h, :orig_w].float()
                else:
                    h_hat = h_hat_padded.float()

                # ------------------------------------------------------------------
                # Step 2: Complex Conversion, Symbol Capacity Check & Resource Mapping
                # ------------------------------------------------------------------
                h_hat_complex = real16_to_complex8(h_hat)       # (B, 8, 14, 52)
                h_noisy_complex = real16_to_complex8(h_noisy)   # (B, 8, 14, 52)

                snr_tensor = torch.full((B, 1), codec_snr_val, device=device, dtype=torch.float32)

                # First slice with expected_num_symbols if H-conditioned encoder needs channel statistics
                if is_h_module:
                    h_hat_sub_pre = slice_channel_resources(
                        h_hat_complex, antenna_idx, t_start, required_T, expected_num_symbols, subcarrier_idx_tensor
                    )
                    h_mean_real = torch.mean(h_hat_sub_pre.real, dim=-1, keepdim=True)
                    h_mean_imag = torch.mean(h_hat_sub_pre.imag, dim=-1, keepdim=True)
                    features = encoder(x_clean, snr_tensor, h_mean_real, h_mean_imag)
                else:
                    features = encoder(x_clean, snr_tensor)

                _, tcn_feat, h_feat, w_feat = features.shape
                f = features.reshape(B, -1)
                num_symbols = f.shape[1] // 2
                T = verify_symbol_capacity(num_symbols)

                # Slice selected antenna, 48 valid subcarriers, and T time slots -> (B, num_symbols)
                if num_symbols != expected_num_symbols or T != required_T:
                    h_true_sub = slice_channel_resources(
                        h_true_complex, antenna_idx, t_start, T, num_symbols, subcarrier_idx_tensor
                    )
                h_hat_sub = slice_channel_resources(
                    h_hat_complex, antenna_idx, t_start, T, num_symbols, subcarrier_idx_tensor
                )
                if eval_baselines:
                    h_noisy_sub = slice_channel_resources(
                        h_noisy_complex, antenna_idx, t_start, T, num_symbols, subcarrier_idx_tensor
                    )

                # ------------------------------------------------------------------
                # Step 3: Wireless Transmission, ZF Equalization & SC Decoding
                # ------------------------------------------------------------------
                z_raw = torch.complex(f[:, :num_symbols], f[:, num_symbols:])  # (B, num_symbols)

                # Power Normalization: E[|z|^2] = 1
                power_z = torch.mean(torch.abs(z_raw) ** 2, dim=1, keepdim=True)
                z = z_raw / torch.sqrt(torch.clamp(power_z, min=1e-12))

                # Wireless channel transmission: y = z * h_true_sub + n
                tx_noise_r = generate_batch_gaussian_noise(
                    shape=(B, num_symbols),
                    noise_seeds=noise_seeds,
                    seed_offset=2002,
                    device=device,
                    dtype=torch.float32,
                ) * sigma_per_real_imag.view(B, 1)
                tx_noise_i = generate_batch_gaussian_noise(
                    shape=(B, num_symbols),
                    noise_seeds=noise_seeds,
                    seed_offset=3003,
                    device=device,
                    dtype=torch.float32,
                ) * sigma_per_real_imag.view(B, 1)
                n = torch.complex(tx_noise_r, tx_noise_i)

                y = z * h_true_sub + n

                # Denoised Channel-based Zero-Forcing (ZF) Equalization
                z_hat = y / (h_hat_sub + 1e-8)

                # Update SNR metrics for Diffusion
                ch_snr_diff = cal_SNR(h_hat_sub, h_true_sub)
                sig_snr_diff = cal_SNR(z_hat, z)
                ch_snr_diff_ratio_sum += torch.sum(ch_snr_diff).to(torch.float64)
                sig_snr_diff_ratio_sum += torch.sum(sig_snr_diff).to(torch.float64)
                total_eval_samples += B

                def _decode_symbols(
                    z_sym: torch.Tensor,
                    ch_ref: Optional[torch.Tensor] = None,
                    dec_snr: torch.Tensor = snr_tensor,
                ) -> torch.Tensor:
                    r_part = torch.nan_to_num(z_sym.real, nan=0.0, posinf=10.0, neginf=-10.0)
                    i_part = torch.nan_to_num(z_sym.imag, nan=0.0, posinf=10.0, neginf=-10.0)

                    if opt.normalize_power:
                        cur_pow = torch.mean(r_part ** 2 + i_part ** 2, dim=1, keepdim=True)
                        expected_pow = 1.0 + 10.0 ** (-dec_snr / 10.0)
                        scale = torch.sqrt(expected_pow / torch.clamp(cur_pow, min=1e-8))
                        r_part = r_part * scale
                        i_part = i_part * scale

                    f_dec = torch.cat([r_part, i_part], dim=1).reshape(B, tcn_feat, h_feat, w_feat)
                    # Clamp before GDN to prevent exponential overflow on deep-fade outliers
                    f_dec = torch.clamp(f_dec, min=-10.0, max=10.0)

                    if is_h_module:
                        ref = ch_ref if ch_ref is not None else h_hat_sub
                        hr_mean = torch.mean(ref.real, dim=-1, keepdim=True)
                        hi_mean = torch.mean(ref.imag, dim=-1, keepdim=True)
                        out_img = decoder(f_dec, dec_snr, hr_mean, hi_mean)
                    else:
                        out_img = decoder(f_dec, dec_snr)

                    out_img = torch.nan_to_num(out_img, nan=0.0, posinf=1.0, neginf=0.0)
                    return torch.clamp(out_img, 0.0, 1.0)

                x_hat_diff = _decode_symbols(z_hat, ch_ref=h_hat_sub)
                psnr_diff_metric.update(x_hat_diff, x_clean)
                msssim_diff_metric.update(x_hat_diff, x_clean)

                # ------------------------------------------------------------------
                # Step 4: Baselines Evaluation (NoDiff, PerfectCSI, Clean)
                # ------------------------------------------------------------------
                if eval_baselines:
                    # 1) No-Diffusion: ZF equalization with noisy channel estimate h_noisy_sub
                    z_hat_nodiff = y / (h_noisy_sub + 1e-8)
                    ch_snr_nodiff = cal_SNR(h_noisy_sub, h_true_sub)
                    sig_snr_nodiff = cal_SNR(z_hat_nodiff, z)
                    ch_snr_nodiff_ratio_sum += torch.sum(ch_snr_nodiff).to(torch.float64)
                    sig_snr_nodiff_ratio_sum += torch.sum(sig_snr_nodiff).to(torch.float64)

                    x_hat_nodiff = _decode_symbols(z_hat_nodiff, ch_ref=h_noisy_sub)
                    psnr_nodiff_metric.update(x_hat_nodiff, x_clean)
                    msssim_nodiff_metric.update(x_hat_nodiff, x_clean)

                    # 2) Perfect CSI: ZF equalization with ground-truth channel h_true_sub
                    z_hat_trueh = y / (h_true_sub + 1e-8)
                    sig_snr_trueh = cal_SNR(z_hat_trueh, z)
                    sig_snr_trueh_ratio_sum += torch.sum(sig_snr_trueh).to(torch.float64)

                    x_hat_trueh = _decode_symbols(z_hat_trueh, ch_ref=h_true_sub)
                    psnr_trueh_metric.update(x_hat_trueh, x_clean)
                    msssim_trueh_metric.update(x_hat_trueh, x_clean)

                    # 3) Clean Latent Upper Bound: noiseless z directly to decoder
                    x_hat_clean = _decode_symbols(z, ch_ref=h_true_sub)
                    psnr_clean_metric.update(x_hat_clean, x_clean)
                    msssim_clean_metric.update(x_hat_clean, x_clean)

                if opt.save_images and saved_grid_images is None:
                    n_vis = min(8, B)
                    vis_orig = x_clean[:n_vis].cpu()
                    vis_diff = x_hat_diff[:n_vis].cpu()
                    if eval_baselines:
                        vis_clean = x_hat_clean[:n_vis].cpu()
                        vis_trueh = x_hat_trueh[:n_vis].cpu()
                        vis_nodiff = x_hat_nodiff[:n_vis].cpu()
                        # Stack rows: [Original, Clean, PerfectCSI, NoDiff, Diff]
                        saved_grid_images = torch.cat([vis_orig, vis_clean, vis_trueh, vis_nodiff, vis_diff], dim=0)
                    else:
                        saved_grid_images = torch.cat([vis_orig, vis_diff], dim=0)

                if dist.get_rank() == 0:
                    sys.stdout.flush()

            if dist.get_rank() == 0:
                tqdm_bar.close()
                sys.stdout.flush()

            # Distributed synchronization for SNR sums
            if dist.get_world_size() > 1:
                torch.distributed.all_reduce(ch_snr_diff_ratio_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(sig_snr_diff_ratio_sum, op=torch.distributed.ReduceOp.SUM)
                if eval_baselines:
                    torch.distributed.all_reduce(ch_snr_nodiff_ratio_sum, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(sig_snr_nodiff_ratio_sum, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(sig_snr_trueh_ratio_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(total_eval_samples, op=torch.distributed.ReduceOp.SUM)

            denom = total_eval_samples.clamp(min=1)
            score_ch_snr_diff = ratio_to_dB((ch_snr_diff_ratio_sum / denom).item())
            score_sig_snr_diff = ratio_to_dB((sig_snr_diff_ratio_sum / denom).item())
            score_psnr_diff = float(psnr_diff_metric.compute().item())
            score_msssim_diff = float(msssim_diff_metric.compute().item())

            row_data: Dict[str, Any] = {
                'SNR_dB': snr_db_val,
                'Ch_SNR_Diff': score_ch_snr_diff,
                'Sig_SNR_Diff': score_sig_snr_diff,
                'PSNR_Diff': score_psnr_diff,
                'MSSSIM_Diff': score_msssim_diff,
            }

            if eval_baselines:
                score_ch_snr_nodiff = ratio_to_dB((ch_snr_nodiff_ratio_sum / denom).item())
                score_sig_snr_nodiff = ratio_to_dB((sig_snr_nodiff_ratio_sum / denom).item())
                score_psnr_nodiff = float(psnr_nodiff_metric.compute().item())
                score_msssim_nodiff = float(msssim_nodiff_metric.compute().item())

                score_sig_snr_trueh = ratio_to_dB((sig_snr_trueh_ratio_sum / denom).item())
                score_psnr_trueh = float(psnr_trueh_metric.compute().item())
                score_msssim_trueh = float(msssim_trueh_metric.compute().item())

                score_psnr_clean = float(psnr_clean_metric.compute().item())
                score_msssim_clean = float(msssim_clean_metric.compute().item())

                row_data.update({
                    'Ch_SNR_NoDiff': score_ch_snr_nodiff,
                    'Sig_SNR_NoDiff': score_sig_snr_nodiff,
                    'PSNR_NoDiff': score_psnr_nodiff,
                    'MSSSIM_NoDiff': score_msssim_nodiff,
                    'Sig_SNR_PerfectCSI': score_sig_snr_trueh,
                    'PSNR_PerfectCSI': score_psnr_trueh,
                    'MSSSIM_PerfectCSI': score_msssim_trueh,
                    'PSNR_Clean': score_psnr_clean,
                    'MSSSIM_Clean': score_msssim_clean,
                })

                dist.print0(
                    f'{snr_db_val:7.1f} | '
                    f'{score_ch_snr_diff:11.4f} | {score_sig_snr_diff:12.4f} | '
                    f'{score_psnr_diff:10.4f} | {score_msssim_diff:12.4f} | '
                    f'{score_ch_snr_nodiff:11.4f} | {score_psnr_nodiff:10.4f} | {score_msssim_nodiff:12.4f} | '
                    f'{score_psnr_trueh:11.4f} | {score_msssim_trueh:13.4f} | '
                    f'{score_psnr_clean:11.4f} | {score_msssim_clean:13.4f}'
                )
            else:
                dist.print0(
                    f'{snr_db_val:7.1f} | '
                    f'{score_ch_snr_diff:11.4f} | {score_sig_snr_diff:12.4f} | '
                    f'{score_psnr_diff:10.4f} | {score_msssim_diff:12.4f}'
                )

            results_table.append(row_data)

            if opt.save_images and saved_grid_images is not None and dist.get_rank() == 0:
                img_dir = os.path.join(os.path.dirname(out_csv), 'sample_images_ch_denoise')
                os.makedirs(img_dir, exist_ok=True)
                grid_path = os.path.join(img_dir, f'comparison_snr_{snr_db_val:04.1f}dB.png')
                n_vis = min(8, opt.batch_size)
                torchvision.utils.save_image(saved_grid_images, grid_path, nrow=n_vis, normalize=False)
                dist.print0(f'Saved sample comparison grid to: {grid_path}')

    # 7. Save CSV and JSON Results
    if dist.get_rank() == 0:
        dist.print0('=' * 90)
        dist.print0(f'Saving evaluation results to:\n- CSV:  {out_csv}\n- JSON: {out_json}')

        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results_table:
                writer.writerow(r)

        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(results_table, f, indent=2)

        dist.print0('Evaluation successfully finished!')

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

