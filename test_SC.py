import os
import sys
import re
import json
import csv
from pathlib import Path
from itertools import cycle
from typing import Optional, Dict, Any, Tuple, List

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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


def dB_to_ratio(dB):
    return 10.0 ** (dB / 10.0)


def ratio_to_dB(ratio):
    if isinstance(ratio, torch.Tensor):
        ratio = ratio.item()
    if ratio <= 0:
        return -100.0
    return float(10.0 * np.log10(ratio))


def cal_SNR(predict: torch.Tensor, truth: torch.Tensor, complex_axis: Optional[int] = None) -> torch.Tensor:
    """denoising_tester.py와 동일한 방식으로 신호 복원 SNR (선형 ratio)을 계산.

    ratio = PS / PN
      PS = sum(|truth|^2)
      PN = sum(|predict - truth|^2)
    """
    if complex_axis is not None:
        assert not predict.is_complex() and not truth.is_complex(), (
            "'complex_axis' is given while dtype is already complex!"
        )
        if complex_axis >= 0:
            complex_axis += 1

        predict_split = torch.chunk(predict, 2, dim=complex_axis)
        truth_split = torch.chunk(truth, 2, dim=complex_axis)

        predict = predict_split[0] + 1j * predict_split[1]
        truth = truth_split[0] + 1j * truth_split[1]

    axis_list = [i for i in range(1, len(predict.shape))]
    ps = torch.sum(torch.abs(truth) ** 2, dim=axis_list)
    pn = torch.sum(torch.abs(predict - truth) ** 2, dim=axis_list)
    pn = torch.clamp(pn, min=1e-12)
    ratio = ps / pn
    return torch.nan_to_num(ratio, nan=1.0, posinf=1e12, neginf=1.0)


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

    # Check if this checkpoint uses AF_Module_H (channel H condition aware)
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

    encoder.to(device).eval()
    decoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    return encoder, decoder, cfg, tcn, is_h_module


def get_cifar10_test_dataset(cifar_data_root: str, using_train_image=False) -> Dataset:
    """CIFAR-10 테스트 데이터셋을 로드."""
    transform = T.Compose([T.ToTensor()])
    train_test_dir_name = 'test'
    if using_train_image:
        train_test_dir_name = 'train'   
    cifar_candidates = [
        os.path.join(cifar_data_root, 'cifar10', train_test_dir_name),
        os.path.join(cifar_data_root, train_test_dir_name),
        os.path.join(str(_WORKSPACE_DIR), 'data', 'cifar10', train_test_dir_name),
        f'./data/cifar10/{train_test_dir_name}',
        os.path.join(str(_ADJSCC_DIR), 'data', 'cifar10', train_test_dir_name),
    ]
    for cand in cifar_candidates:
        if os.path.exists(cand) and os.path.isdir(cand):
            from torchvision.datasets import ImageFolder
            dist.print0(f'Using CIFAR-10 test set from directory: {cand}')
            return ImageFolder(cand, transform=transform)

    dist.print0(f'Loading torchvision CIFAR-10 test set (root={cifar_data_root})...')
    try:
        from torchvision.datasets import CIFAR10
        return CIFAR10(root=cifar_data_root, train=False, download=False, transform=transform)
    except Exception:
        from torchvision.datasets import CIFAR10
        return CIFAR10(root=cifar_data_root, train=False, download=True, transform=transform)


class PairedSCTestDataset(Dataset):
    """
    CIFAR-10 테스트 이미지와 RENEW RF 채널을 1:1 결정론적으로 페어링하는 데이터셋.
    """
    def __init__(self, image_dataset: Dataset, renew_channel_dataset: Dataset, max_samples: Optional[int] = None):
        self.image_dataset = image_dataset
        self.renew_dataset = renew_channel_dataset
        self.n_images = len(image_dataset)
        self.n_channels = len(renew_channel_dataset)

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

        # Cyclic channel indexing
        ch_idx = idx % self.n_channels
        ch_item = self.renew_dataset[ch_idx]
        raw_h = ch_item.get('image', ch_item.get('clean_channel'))

        return {
            'image': img.float(),
            'channel': raw_h.float() if isinstance(raw_h, torch.Tensor) else torch.from_numpy(raw_h).float(),
            'idx': idx
        }


#----------------------------------------------------------------------------
# CLI Command Setup

@click.command()
@click.option('--network_pkl',              help='Network pickle filename (.pkl)', metavar='PATH', type=str, required=True)
@click.option('--config_json',              help='Network training options JSON filename', metavar='PATH', type=str, default=None)
@click.option('--adjscc_config',            help='Path to ADJSCC config JSON/YAML file', type=str, default='misc/ADJSCC/config_example.json')
@click.option('--adjscc_model_path',        help='Explicit path to ADJSCC checkpoint .pt', type=str, default=None)
@click.option('--data',                     help='Path to RENEW RF CSI dataset npz directory', metavar='DIR', type=str, required=True)
@click.option('--data_norm',                help='Data normalization value for the RF data (default: auto from config)', type=float, default=None)
@click.option('--cifar_data',               help='Root path for CIFAR-10 test data', type=str, default='./data')
@click.option('--using_train_image',        help='Use train image instead of test image', is_flag=True)
@click.option('--data_keep_ratio',          help='Data keep ratio for RENEW channel dataset', type=float, default=1.0)
@click.option('--must_contain',             help='Filter RENEW filenames to contain', type=str, default=None)
@click.option('--must_not_contain',         help='Filter RENEW filenames not to contain', type=str, default=None)
@click.option('--flip_dataset',             help='Use held-out/test split of RENEW data', is_flag=True)
@click.option('--test_SNR_range', 'test_SNR_range', help='SNR range to test (e.g. 0,20)', type=str, default='0,20')
@click.option('--test_SNR_step', 'test_SNR_step',   help='SNR step size (dB)', type=float, default=2.0)
@click.option('--fixed_codec_snr',          help='Fixed SNR (dB) conditioned to Encoder/Decoder', type=float, default=None)
@click.option('--batch_size',               help='Batch size for evaluation', type=int, default=64)
@click.option('--max_samples',              help='Maximum test samples (default: full CIFAR-10 test set)', type=int, default=None)
@click.option('--steps', 'num_steps',      help='Number of EDM sampling steps', type=click.IntRange(min=1), default=18)
@click.option('--sigma_min',               help='Lowest noise level', type=float, default=None)
@click.option('--sigma_max',               help='Highest noise level', type=float, default=None)
@click.option('--rho',                     help='Time step exponent', type=float, default=7.0)
@click.option('--S_churn', 'S_churn',     help='Stochasticity strength', type=float, default=0.0)
@click.option('--S_min', 'S_min',         help='Stochasticity min noise level', type=float, default=0.0)
@click.option('--S_max', 'S_max',         help='Stochasticity max noise level', type=float, default=float('inf'))
@click.option('--S_noise', 'S_noise',     help='Stochasticity noise inflation', type=float, default=1.0)
@click.option('--bf16',                    help='Enable bfloat16 precision for diffusion sampling', is_flag=True)
@click.option('--allow_tf32',              help='Allow TF32 matmul and cudnn', is_flag=True)
@click.option('--no_baselines',            help='Disable baseline evaluations (No-Diffusion and Clean-Latent)', is_flag=True)
@click.option('--save_images',             help='Save visual comparison image grids (.png) for each SNR', is_flag=True)
@click.option('--output',                   help='Output CSV filename or directory', type=str, default=None)
@click.option('--seed',                     help='Random seed for reproducibility', type=int, default=11454)
@click.option('--force_no_nn_scheduler',   help='Force noise scheduling to operate in no_nn_scheduler mode', is_flag=True)
@click.option('--scheduler_mode',           help='Override scheduler mode for the network', type=str, default=None)
@click.option('--padding_power2',          help='Pad channel dimensions to power of 2 for U-Net', is_flag=True)
@click.option('--channel_type',            help='Override channel type (slow_fading_eq or slow_fading)', type=str, default=None)
@click.option('--slow_fading_eq',          help='Explicitly set slow_fading_eq flag', type=bool, default=None)
def main(**kwargs):
    # Enable line buffering so tqdm updates immediately in torchrun / multi-GPU pipes
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    dist.init()
    device = torch.device('cuda')
    opt = dnnlib.EasyDict(kwargs)

    # Distributed barrier
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if opt.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # 1. Load Diffusion Model
    dist.print0(f'Loading diffusion network from "{opt.network_pkl}"...')
    data = save_utils.load_pkl(opt.network_pkl, verbose=(dist.get_rank() == 0))
    net = data['ema'].to(device)
    if opt.bf16:
        if hasattr(net, 'use_fp16'):
            net.use_fp16 = True
        if hasattr(getattr(net, 'module', None), 'use_fp16'):
            net.module.use_fp16 = True

    # 2. Load Config JSON
    if opt.config_json is not None:
        config_filepath = opt.config_json
    else:
        config_filepath = os.path.join(os.path.dirname(opt.network_pkl), 'training_options.json')

    if os.path.isfile(config_filepath):
        dist.print0(f'Loading training options from "{config_filepath}"...')
        with open(config_filepath, 'r', encoding='utf-8') as f:
            train_opts = json.load(f)
    else:
        dist.print0('Warning: training_options.json not found. Using default arguments.')
        train_opts = {}

    # Determine channel configuration
    ds_kwargs = train_opts.get('dataset_kwargs', {})
    inferred_channel_type = ds_kwargs.get('channel_type', 'slow_fading_eq')
    inferred_slow_fading_eq = ds_kwargs.get('slow_fading_eq', (inferred_channel_type == 'slow_fading_eq'))

    channel_type = opt.channel_type if opt.channel_type is not None else inferred_channel_type
    slow_fading_eq = opt.slow_fading_eq if opt.slow_fading_eq is not None else (
        opt.channel_type == 'slow_fading_eq' if opt.channel_type is not None else inferred_slow_fading_eq
    )

    # Data normalization value
    inferred_data_norm = float(ds_kwargs.get('normalize_value', 1.0))
    data_norm = float(opt.data_norm) if opt.data_norm is not None else inferred_data_norm
    dist.print0(f'Channel mode: channel_type="{channel_type}", slow_fading_eq={slow_fading_eq}, data_norm={data_norm}')

    # Noise Scheduler Mode
    target_mode = (
        opt.scheduler_mode
        if opt.scheduler_mode is not None
        else train_opts.get('network_kwargs', {}).get(
            'scheduler_mode',
            train_opts.get('scheduler_mode', getattr(net, 'scheduler_mode', 'using_sigma_t_n'))
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

    if opt.force_no_nn_scheduler:
        dist.print0('Forcing noise scheduler to "no_nn_scheduler" mode.')
        net.scheduler_mode = 'no_nn_scheduler'
        if hasattr(net, 'noise_scheduler'):
            net.noise_scheduler.scheduler_mode = 'no_nn_scheduler'

    # Other ranks follow
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # 3. Load ADJSCC Models (Encoder / Decoder)
    encoder, decoder, adjscc_cfg, tcn, is_h_module = load_adjscc_models(
        adjscc_config_path=opt.adjscc_config,
        explicit_model_path=opt.adjscc_model_path,
        device=device
    )

    # Calculate T (time steps)
    T_steps = calculate_sc_t_from_config(adjscc_cfg, num_subcarriers=48)
    dist.print0(f'ADJSCC transmission time steps T = {T_steps} (subcarriers=48)')

    # 4. Load Datasets
    dist.print0('Initializing CIFAR-10 test dataset...')
    cifar_test_ds = get_cifar10_test_dataset(opt.cifar_data)

    dist.print0(f'Initializing RENEW RF Channel dataset from "{opt.data}" (normalize_value={data_norm})...')
    renew_channel_ds = RenewChannelDataset(
        renew_path=opt.data,
        adjscc_config=adjscc_cfg,
        T=T_steps,
        seed=opt.seed,
        must_contain=opt.must_contain,
        must_not_contain=opt.must_not_contain,
        dataset_keep_percentage=opt.data_keep_ratio,
        flip_keep_dataset=opt.flip_dataset,
        normalize_value=data_norm,
    )

    paired_test_ds = PairedSCTestDataset(
        image_dataset=cifar_test_ds,
        renew_channel_dataset=renew_channel_ds,
        max_samples=opt.max_samples
    )
    dist.print0(f'Total test samples to evaluate: {len(paired_test_ds)}')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        paired_test_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )
    dataloader = DataLoader(
        dataset=paired_test_ds,
        sampler=dist_sampler,
        batch_size=opt.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # 5. Setup Metrics (torchmetrics)
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
        psnr_clean_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        msssim_clean_metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=3, betas=(0.0448, 0.2856, 0.3001), normalize='relu'
        ).to(device)

    # 6. Parse SNR Steps
    snr_range = [float(x) for x in opt.test_SNR_range.split(',')]
    snr_steps = np.arange(snr_range[0], snr_range[1] + 1e-5, opt.test_SNR_step)
    dist.print0(f'Target SNR range: {snr_steps.tolist()} dB')

    # Sampler kwargs
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

    # Setup Output Paths
    if opt.output is not None:
        out_csv = opt.output if opt.output.endswith('.csv') else os.path.join(opt.output, 'eval_sc_results.csv')
    else:
        out_dir = os.path.dirname(opt.network_pkl)
        out_csv = os.path.join(out_dir, 'eval_sc_results.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out_json = out_csv.replace('.csv', '.json')

    results_table = []
    headers = ['SNR_dB', 'Signal_SNR_Diffusion', 'PSNR_Diffusion', 'MSSSIM_Diffusion']
    if eval_baselines:
        headers.extend(['Signal_SNR_NoDiff', 'PSNR_NoDiff', 'MSSSIM_NoDiff', 'PSNR_Clean', 'MSSSIM_Clean'])

    # Print Table Header
    dist.print0('\n' + '=' * 125)
    if eval_baselines:
        dist.print0(
            f'{"SNR (dB)":>9} | '
            f'{"Sig SNR (Diff)":>14} | {"PSNR (Diff)":>11} | {"MS-SSIM (Diff)":>14} | '
            f'{"Sig SNR (NoDiff)":>16} | {"PSNR (NoDiff)":>13} | {"MS-SSIM (NoDiff)":>16} | '
            f'{"PSNR (Clean)":>12} | {"MS-SSIM (Clean)":>15}'
        )
        dist.print0('-' * 125)
    else:
        dist.print0(
            f'{"SNR (dB)":>9} | '
            f'{"Sig SNR (Diff)":>14} | {"PSNR (Diff)":>11} | {"MS-SSIM (Diff)":>14}'
        )
        dist.print0('-' * 60)

    # 7. Main Evaluation Loop over SNR Steps
    with torch.inference_mode(True):
        for snr_db in snr_steps:
            snr_db_val = float(snr_db)
            codec_snr_val = float(opt.fixed_codec_snr) if opt.fixed_codec_snr is not None else snr_db_val

            # Reset metrics
            psnr_diff_metric.reset()
            msssim_diff_metric.reset()
            if eval_baselines:
                psnr_nodiff_metric.reset()
                msssim_nodiff_metric.reset()
                psnr_clean_metric.reset()
                msssim_clean_metric.reset()

            snr_diff_ratio_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            snr_nodiff_ratio_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            total_eval_samples = torch.tensor(0, device=device, dtype=torch.long)

            saved_grid_images = None

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
                x_clean = batch['image'].to(device)     # (B, 3, 32, 32)
                raw_h = batch['channel'].to(device)     # (B, 2, T, 48)
                B = x_clean.shape[0]

                # Convert H to complex
                if raw_h.ndim == 4:
                    h_complex = torch.complex(raw_h[:, 0].float(), raw_h[:, 1].float())
                else:
                    h_complex = raw_h.to(torch.complex64)
                B, T, C = h_complex.shape

                # Codec SNR Tensor
                snr_tensor = torch.full((B, 1), codec_snr_val, device=device, dtype=torch.float32)

                # 1) ADJSCC Encoder Forward
                if is_h_module:
                    h_mean_real = torch.mean(h_complex.real, dim=(-2, -1), keepdim=True).view(B, 1)
                    h_mean_imag = torch.mean(h_complex.imag, dim=(-2, -1), keepdim=True).view(B, 1)
                    features = encoder(x_clean, snr_tensor, h_mean_real, h_mean_imag)
                else:
                    features = encoder(x_clean, snr_tensor)

                # 2) Complex Conversion & Power Normalization
                f = features.reshape(B, -1)
                dim_z = f.shape[1] // 2
                z_in = torch.complex(f[:, :dim_z], f[:, dim_z:])

                # Normalize symbol power
                norm_factor = torch.sum(z_in.real ** 2 + z_in.imag ** 2, dim=1, keepdim=True)
                scale = torch.sqrt(dim_z / torch.clamp(norm_factor, min=1e-8))
                z_in = z_in * scale

                # 3) Latent Shape Matching to (B, T, C)
                target_numel = T * C
                if dim_z == target_numel:
                    z_matched = z_in.reshape(B, T, C)
                else:
                    real_1d = z_in.real.unsqueeze(1)
                    imag_1d = z_in.imag.unsqueeze(1)
                    real_interp = F.interpolate(real_1d, size=target_numel, mode='linear', align_corners=False).squeeze(1)
                    imag_interp = F.interpolate(imag_1d, size=target_numel, mode='linear', align_corners=False).squeeze(1)
                    z_matched = torch.complex(real_interp, imag_interp).reshape(B, T, C)

                power_z = torch.mean(torch.abs(z_matched) ** 2, dim=(-2, -1), keepdim=True)
                z_matched = z_matched / torch.sqrt(torch.clamp(power_z, min=1e-12))

                # 4) Wireless Channel Transmission & Noise Synthesis
                ratio_snr = 10.0 ** (snr_db_val / 10.0)

                if slow_fading_eq:
                    clean_target_signal = z_matched
                    sig_pow = torch.ones((B, 1, 1), device=device, dtype=torch.float32)
                else:
                    rx_noiseless = h_complex * z_matched
                    clean_target_signal = rx_noiseless
                    sig_pow = torch.mean(torch.abs(rx_noiseless) ** 2, dim=(-2, -1), keepdim=True)

                # Complex Gaussian noise standard deviation per real/imag dimension
                eff_sigma = torch.sqrt(sig_pow / (ratio_snr * 2.0)).float()

                noise_r = torch.randn((B, T, C), device=device, dtype=torch.float32) * eff_sigma
                noise_i = torch.randn((B, T, C), device=device, dtype=torch.float32) * eff_sigma
                rx_noise = torch.complex(noise_r, noise_i)

                eps = 1e-6
                h_mag = torch.abs(h_complex)
                h_safe = torch.where(h_mag < eps, eps * torch.exp(1j * torch.angle(h_complex)), h_complex)

                if slow_fading_eq:
                    eq_noise = rx_noise / h_safe
                    rx_signal = clean_target_signal + eq_noise
                    raw_sigma = (torch.sqrt(eff_sigma ** 2 * 2.0) / torch.clamp(h_mag, min=eps)).float()
                else:
                    rx_signal = clean_target_signal + rx_noise
                    raw_sigma = (torch.sqrt(eff_sigma ** 2 * 2.0)).expand(B, T, C).float()

                # Sigma reduction (mean sigma across subcarriers & time)
                input_sigma = torch.mean(raw_sigma, dim=(-2, -1)).float()

                # 5) Diffusion EDM Sampler Denoising
                rx_input = torch.stack([rx_signal.real, rx_signal.imag], dim=1).float()

                if opt.padding_power2:
                    padded_signal, _, orig_shape = pad_collate_fn(rx_input)
                    target_shape = (1,) + tuple(padded_signal.shape[1:])
                    in_orig_shape = torch.tensor(tuple(orig_shape[0])).unsqueeze(0)
                    padding_mask = padding_mask_from_original_shape(in_orig_shape, target_shape)
                else:
                    padded_signal = rx_input
                    padding_mask = 1

                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=opt.bf16):
                    denoised_signal, _ = edm_sampler(
                        net,
                        latents=padded_signal,
                        sigma_max=input_sigma,
                        padding_mask=padding_mask,
                        latents_already_noisy=True,
                        **sampler_kwargs
                    )

                if opt.padding_power2:
                    denoised_signal = denoised_signal[:, :, :T, :C]

                denoised_signal = denoised_signal.float()
                denoised_target_signal = torch.complex(denoised_signal[:, 0], denoised_signal[:, 1])
                z_hat_complex = denoised_target_signal

                # Equalize if regular slow_fading (MMSE regularized to avoid deep fade explosion)
                if not slow_fading_eq:
                    h_mag2 = torch.abs(h_complex) ** 2
                    eps_eq = 1e-4
                    z_hat_complex = (z_hat_complex * torch.conj(h_complex)) / (h_mag2 + eps_eq)

                # 6) Inverse Shape Matching to (B, tcn, 8, 8)
                def _inverse_match_and_decode(z_tensor: torch.Tensor) -> torch.Tensor:
                    z_tensor = torch.nan_to_num(z_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
                    if target_numel == dim_z:
                        z_unmatch = z_tensor.reshape(B, dim_z)
                    else:
                        r_flat = z_tensor.real.reshape(B, -1).unsqueeze(1)
                        i_flat = z_tensor.imag.reshape(B, -1).unsqueeze(1)
                        r_inv = F.interpolate(r_flat, size=dim_z, mode='linear', align_corners=False).squeeze(1)
                        i_inv = F.interpolate(i_flat, size=dim_z, mode='linear', align_corners=False).squeeze(1)
                        z_unmatch = torch.complex(r_inv, i_inv)

                    f_dec = torch.cat([z_unmatch.real, z_unmatch.imag], dim=1).reshape(B, tcn, 8, 8)
                    # Clamp latent symbols before GDN to prevent exponential overflow in decoder
                    f_dec = torch.clamp(f_dec, min=-10.0, max=10.0)
                    if is_h_module:
                        out_img = decoder(f_dec, snr_tensor, h_mean_real, h_mean_imag)
                    else:
                        out_img = decoder(f_dec, snr_tensor)
                    out_img = torch.nan_to_num(out_img, nan=0.0, posinf=1.0, neginf=0.0)
                    return torch.clamp(out_img, 0.0, 1.0)

                # Decode Diffusion Reconstructed Image
                x_hat_diff = _inverse_match_and_decode(z_hat_complex)

                # Update Diffusion Metrics
                psnr_diff_metric.update(x_hat_diff, x_clean)
                msssim_diff_metric.update(x_hat_diff, x_clean)

                # Signal SNR for Diffusion (diffusion model output vs clean target signal)
                diff_snr_ratio = cal_SNR(denoised_target_signal, clean_target_signal)
                snr_diff_ratio_sum += torch.sum(diff_snr_ratio).to(torch.float64)
                total_eval_samples += B

                # 7) Baselines Evaluation
                if eval_baselines:
                    # No Diffusion: for slow_fading, use MMSE equalization to avoid deep fade noise amplification
                    if slow_fading_eq:
                        z_nodiff = rx_signal
                    else:
                        h_mag2 = torch.abs(h_complex) ** 2
                        noise_var = torch.mean(torch.abs(rx_noise) ** 2, dim=(-2, -1), keepdim=True)
                        z_nodiff = (rx_signal * torch.conj(h_complex)) / (h_mag2 + noise_var)

                    # Signal SNR for NoDiff (input noisy signal vs clean target signal)
                    nodiff_snr_ratio = cal_SNR(rx_signal, clean_target_signal)
                    snr_nodiff_ratio_sum += torch.sum(nodiff_snr_ratio).to(torch.float64)

                    x_hat_nodiff = _inverse_match_and_decode(z_nodiff)
                    psnr_nodiff_metric.update(x_hat_nodiff, x_clean)
                    msssim_nodiff_metric.update(x_hat_nodiff, x_clean)

                    # Clean Latent Upper Bound
                    x_hat_clean = _inverse_match_and_decode(z_matched)
                    psnr_clean_metric.update(x_hat_clean, x_clean)
                    msssim_clean_metric.update(x_hat_clean, x_clean)

                # Save sample image comparison tensor from the last batch
                if opt.save_images and saved_grid_images is None:
                    n_vis = min(8, B)
                    vis_orig = x_clean[:n_vis].cpu()
                    vis_diff = x_hat_diff[:n_vis].cpu()
                    if eval_baselines:
                        vis_clean = x_hat_clean[:n_vis].cpu()
                        vis_nodiff = x_hat_nodiff[:n_vis].cpu()
                        # Stack: [Original, Clean, NoDiff, Diff]
                        saved_grid_images = torch.cat([vis_orig, vis_clean, vis_nodiff, vis_diff], dim=0)
                    else:
                        saved_grid_images = torch.cat([vis_orig, vis_diff], dim=0)

                if dist.get_rank() == 0:
                    sys.stdout.flush()

            if dist.get_rank() == 0:
                tqdm_bar.close()
                sys.stdout.flush()

            # Distributed synchronization for Signal SNR
            if dist.get_world_size() > 1:
                torch.distributed.all_reduce(snr_diff_ratio_sum, op=torch.distributed.ReduceOp.SUM)
                if eval_baselines:
                    torch.distributed.all_reduce(snr_nodiff_ratio_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(total_eval_samples, op=torch.distributed.ReduceOp.SUM)

            mean_ratio_diff = (snr_diff_ratio_sum / total_eval_samples.clamp(min=1)).item()
            score_snr_diff = ratio_to_dB(mean_ratio_diff)

            # Compute Average Image Metrics for Current SNR
            score_psnr_diff = float(psnr_diff_metric.compute().item())
            score_msssim_diff = float(msssim_diff_metric.compute().item())

            row_data = {
                'SNR_dB': snr_db_val,
                'Signal_SNR_Diffusion': score_snr_diff,
                'PSNR_Diffusion': score_psnr_diff,
                'MSSSIM_Diffusion': score_msssim_diff,
            }

            if eval_baselines:
                mean_ratio_nodiff = (snr_nodiff_ratio_sum / total_eval_samples.clamp(min=1)).item()
                score_snr_nodiff = ratio_to_dB(mean_ratio_nodiff)

                score_psnr_nodiff = float(psnr_nodiff_metric.compute().item())
                score_msssim_nodiff = float(msssim_nodiff_metric.compute().item())
                score_psnr_clean = float(psnr_clean_metric.compute().item())
                score_msssim_clean = float(msssim_clean_metric.compute().item())

                row_data.update({
                    'Signal_SNR_NoDiff': score_snr_nodiff,
                    'PSNR_NoDiff': score_psnr_nodiff,
                    'MSSSIM_NoDiff': score_msssim_nodiff,
                    'PSNR_Clean': score_psnr_clean,
                    'MSSSIM_Clean': score_msssim_clean,
                })

                dist.print0(
                    f'{snr_db_val:9.1f} | '
                    f'{score_snr_diff:14.4f} | {score_psnr_diff:11.4f} | {score_msssim_diff:14.4f} | '
                    f'{score_snr_nodiff:16.4f} | {score_psnr_nodiff:13.4f} | {score_msssim_nodiff:16.4f} | '
                    f'{score_psnr_clean:12.4f} | {score_msssim_clean:15.4f}'
                )
            else:
                dist.print0(
                    f'{snr_db_val:9.1f} | '
                    f'{score_snr_diff:14.4f} | {score_psnr_diff:11.4f} | {score_msssim_diff:14.4f}'
                )

            results_table.append(row_data)

            # Save Visual Image Grid (.png)
            if opt.save_images and saved_grid_images is not None and dist.get_rank() == 0:
                img_dir = os.path.join(os.path.dirname(out_csv), 'sample_images')
                os.makedirs(img_dir, exist_ok=True)
                grid_path = os.path.join(img_dir, f'comparison_snr_{snr_db_val:04.1f}dB.png')
                n_rows = 4 if eval_baselines else 2
                torchvision.utils.save_image(saved_grid_images, grid_path, nrow=n_vis, normalize=False)
                dist.print0(f'Saved sample comparison grid to: {grid_path}')

    # 8. Save CSV and JSON Results
    if dist.get_rank() == 0:
        dist.print0('=' * 80)
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
