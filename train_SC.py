# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model for Semantic Communication (SC)."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop_SC as training_loop
from training.dataset import rf_augmentation_collate_fn
from training.semantic_comm_dataset import RenewChannelDataset, RenewSemanticCommunicationDataset
import ambient_utils
import warnings
import wandb
import string
import random
import copy

warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--task',          help='Current task',         metavar='STR',                        type=str, default='RENEW_SC', show_default=True)
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ddpmpp_256|ddpmpp_192|ddpmpp_64|rf_dit|rf_dit_large|rf_dit_small|rf_dit_real|rf_dit_real_large|rf_dit_real_deep|rf_dit_real_small',          type=str, default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm|edm_dynamic|edm_boosted_sigma|edm_loss_scaling_test|edm_with_scheduler|edm_with_latent_z',       type=click.Choice(['vp', 've', 'edm', 'edm_dynamic', 'edm_boosted_sigma', 'edm_loss_scaling_test', 'edm_with_scheduler', 'edm_with_latent_z']), default='edm', show_default=True)
@click.option('--real_precond',  help='Preconditioning & loss function', metavar='edm|edm_c_skip|edm_input_scaling_test|edm_with_scheduler',       type=click.Choice(['edm', 'edm_c_skip', 'edm_input_scaling_test', 'edm_with_scheduler']), default='edm', show_default=True)
@click.option('--no_asm',        help='Force not to use ASM Loss',                                  is_flag=True)
@click.option('--must_contain',  help='Dataset name should contain', metavar='STR',                 type=str, default=None, show_default=True)
@click.option('--must_not_contain',help='Dataset name should not contain', metavar='STR',            type=str, default=None, show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=150, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=2e-4, show_default=True)
@click.option('--weight_decay',  help='Weight decay', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=False), default=0.0, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.05, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)
@click.option('--label_dropout', help='Label dropout probability for classifier-free guidance', metavar='FLOAT',  type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option('--attn_scale',    help='Attention scale (temperature scaling factor)', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--grad_clip',     help='Gradient clipping', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1e8, show_default=True)
@click.option('--lr_rampup_kimg',     help='LR Rampup in kimg', metavar='INT', type=click.IntRange(min=0, min_open=True), default=10000, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--allow_tf32',    help='allowing tf32', metavar='BOOL',                              is_flag=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--compile', 'compile_network', help='Enable torch.compile for the training network', is_flag=True, default=False)
@click.option('--compile_mode',  help='torch.compile mode', metavar='default|reduce-overhead|max-autotune', type=click.Choice(['default', 'reduce-overhead', 'max-autotune']), default='default', show_default=True)

# I/O-related.
@click.option("--expr_id",      help="Experiment ID", type=str, default="test")
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('--resume_options',help='Resume from previous training options', metavar='JSON',      type=str, default=None)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)
@click.option('--temp_save',     help='Save intermediate results.',                                is_flag=True)

# RF dataset related
@click.option('--view_as_complex', help='Whether to view the data as complex numbers.', type=bool, default=False, show_default=True)
@click.option('--data_norm', help='Data normalization value for the RF data.', type=float, default=1.0, show_default=True)
@click.option('--flip_aug_ratio', help='Ratio of flip augmentation.', type=float, default=0.0, show_default=True)
@click.option('--phase_shift_aug_ratio', help='Ratio of phase shift augmentation.', type=float, default=0.0, show_default=True)

# Scaling laws related
@click.option("--corruption_probability", help="Controls what percentage of images should be corrupted.", type=float, default=0.0)
@click.option("--sigma", help="How much noise to add to the corrupted images.", type=float, default=0.0)
@click.option('--dataset_keep_percentage', help='Limit training samples.', type=float, default=1.0, show_default=True)
@click.option('--additive_noise_sigma', help='Standard deviation of the additive noise to be added to the clean images during corruption.', type=float, default=0.0, show_default=True)
@click.option('--multiply_noise_sigma', help='Multiplying factor of the noise to be added to the clean images during corruption.', type=float, default=1.0, show_default=True)
@click.option('--dynamic_noise_sigma', help='Wheteher to use mean noise or not.', is_flag=True, default=False)
@click.option('--not_dynamic_noise_sigma_model', help='The model used to predict with dynamic noise sigma for each sample.', is_flag=True, default=False)
@click.option('--noise_mean_alter_way', help='Whether to use alternative way to add noise, which is to directly add noise with the given sigma without multiplying with the clean image.', is_flag=True, default=False)
@click.option('--only_additive_noise', help='Whether to only use additive noise for corruption without natural noise.', is_flag=True)
@click.option('--snr_range', help='Target SNR range in dB (e.g. "5,20") for natural noise corruption. Overrides additive_noise_sigma and multiply_noise_sigma if set.', type=str, default=None, show_default=True)

# Semantic Communication (RENEW_SC) related
@click.option('--adjscc_config', help='Path to ADJSCC config JSON/YAML file for RENEW_SC.', type=str, default='misc/ADJSCC/config_example.json', show_default=True)
@click.option('--adjscc_model_path', help='Explicit path to ADJSCC checkpoint .pt file for RENEW_SC (overrides config).', type=str, default=None, show_default=True)
@click.option('--image_data', help='Path to image dataset directory for RENEW_SC (None for CIFAR-10).', type=str, default=None, show_default=True)
@click.option('--sc_t', help='Number of time steps T for RENEW_SC (None to auto-calculate from adjscc_config image size and compression level).', type=int, default=None, show_default=True)
@click.option('--sc_dataset_len', help='Fixed dataset length for RENEW_SC (None for default).', type=int, default=None, show_default=True)
@click.option('--matching_mode', help='Latent to RF channel matching mode for RENEW_SC.', type=click.Choice(['interpolate', 'pad_or_slice', 'repeat']), default='interpolate', show_default=True)
@click.option('--sc_channel_type', help='Channel type for RENEW_SC: "slow_fading" (standard Y = H*Z + N) or "slow_fading_eq" (Zero-Forcing equalization Y/H = Z + N/H with dimensional noise).', type=click.Choice(['slow_fading', 'slow_fading_eq']), default='slow_fading', show_default=True)
@click.option('--slow_fading_eq', help='Shortcut flag to enable slow_fading_eq channel equalization for RENEW_SC.', is_flag=True, default=False)

# Consistency params
@click.option("--consistency_batch_size", help="Batch size for the consistency loss.", type=int, default=32)
@click.option("--with_weight", help="Whether to use weight in the consistency loss.", type=bool, default=False)
@click.option("--with_grad", help="Whether to use gradient in the consistency loss.", type=bool, default=True)
@click.option("--num_consistency_steps", help="Number of steps for the consistency loss.", type=int, default=6)
@click.option("--num_primes", help="Number of primes for the consistency loss.", type=int, default=6)
@click.option("--consistency_coeff", help="Coefficient for the consistency loss.", type=float, default=0.0)

# EDM Loss params
@click.option('--p_mean',            help='P_mean parameter for EDM Loss', metavar='FLOAT', type=float, default=-1.2, show_default=True)
@click.option('--p_std',             help='P_std parameter for EDM Loss', metavar='FLOAT', type=float, default=1.2, show_default=True)
@click.option('--sigma_data',        help='sigma_data parameter for EDM Loss and Precond', metavar='FLOAT', type=float, default=0.5, show_default=True)
@click.option('--sigma_input_scale', help='sigma_input_scale parameter for EDMPrecond_input_scaling_test', metavar='FLOAT', type=float, default=0.5, show_default=True)
@click.option('--sigma_loss_scaling', help='sigma_loss_scaling parameter for EDMLoss_loss_scaling_test', metavar='FLOAT', type=float, default=0.5, show_default=True)
@click.option('--m_dim',             help='Latent z dimension m for scheduler', metavar='INT', type=int, default=50, show_default=True)
@click.option('--k_top',             help='Top-k selection k for scheduler', metavar='INT', type=int, default=15, show_default=True)
@click.option('--rho',               help='Rho parameter for scheduler', metavar='FLOAT', type=float, default=7.0, show_default=True)
@click.option('--kl_coeff',          help='KL loss coefficient for EDMLoss_with_scheduler or EDMLoss_with_latent_z', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--scheduler_mode',    help='Scheduler mode for edm_with_scheduler', metavar='using_sigma_t_n|no_sigma_t_n|no_nn_scheduler|using_sigma_t_n_wo_z|no_nn_scheduler_with_z|only_above_sigma_t_n', type=click.Choice(['using_sigma_t_n', 'no_sigma_t_n', 'no_nn_scheduler', 'using_sigma_t_n_wo_z', 'no_nn_scheduler_with_z', 'only_above_sigma_t_n']), default='using_sigma_t_n', show_default=True)

# Validation params
@click.option('--validation_interval', help='How often to run validation. If -1, no validation will be run', metavar='tick', type=click.IntRange(min=-1), default=50, show_default=True)
@click.option('--validation_iterations', help='Number of iterations to run for validation.', metavar='INT', type=click.IntRange(min=1), default=5, show_default=True)
@click.option('--validation_batch_size', help='Batch size for validation.', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--validation_data', help='Path to the validation data. If not specified, portion of the training data will be used.', metavar='ZIP|DIR', type=str, default=None)

# Wandb related
@click.option('--wandb', help='Use wandb to log training progress',  type=bool, default=True)
@click.option('--wandb_group', help='Wandb group name',                   type=str, default='Test')
@click.option('--debug_test', help='Whether to run a quick test with 1 batch to verify the training loop works.', is_flag=True)

# RF_DiT architecture options
@click.option('--num_dit_blocks', help='Number of DiT blocks for RF_DiT [default: 5]', type=int, default=None)
@click.option('--dit_hidden_dim', help='Hidden dimension of DiT blocks for RF_DiT [default: 256]', type=int, default=None)
@click.option('--dit_num_heads', help='Number of attention heads for DiT blocks [default: 8]', type=int, default=None)
@click.option('--dit_mlp_ratio', help='MLP expansion ratio for DiT blocks [default: 4.0]', type=float, default=None)
@click.option('--num_blocks_per_stage', help='Number of residual blocks per stage in RF_DiT enc/dec [default: 4]', type=int, default=None)
@click.option('--causal_conv', help='Enable causal 2D conv along time dimension for RF_DiT [default: True]', type=bool, default=None)
@click.option('--pos_embed_type', help='Positional embedding mode for RF_DiT (learnable_2d, sincos_2d, hybrid_1d_t_sincos_c_learnable)', type=click.Choice(['learnable_2d', 'sincos_2d', 'hybrid_1d_t_sincos_c_learnable']), default=None)

def main(**kwargs):
    """Train diffusion-based generative model for Semantic Communication (SC).

    Examples:

    \b
    # Train DDPM++ model for Semantic Communication (RENEW_SC)
    torchrun --standalone --nproc_per_node=1 train_SC.py --outdir=training-runs \\
        --data=data/RENEW/splited/ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS \\
        --adjscc_config=misc/ADJSCC/config_example.json --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')

    dist.init()
    
    # Initialize config dict.
    c = dnnlib.EasyDict()

    if opts.debug_test:
        opts.wandb = False
        opts.expr_id = 'debug_test'
        opts.wandb_group = 'debug'
        opts.outdir = 'debug_test'
        opts.nosubdir = True
        opts.validation_interval = 1
        opts.validation_iterations = 1

    if dist.get_rank() == 0 and opts.resume_options is not None and opts.resume is not None:
        with open(opts.resume_options, 'r') as f:
            resume_options = json.load(f)
    else:
        resume_options = None

    wandb_id = None
    if dist.get_rank() == 0 and opts.wandb:
        if resume_options is not None:
            if 'wandb_id' in resume_options:
                wandb_id = resume_options['wandb_id']
                dist.print0(f"Resumed Wandb run with ID: {wandb_id}")
            else:
                dist.print0("No Wandb ID found in resume options, starting a new Wandb run.")

        wandb_run = wandb.init(project="ambient_rf", 
                                config=opts, name=opts.expr_id,
                                group=opts.wandb_group,
                                dir=opts.outdir,
                                id=wandb_id, resume="allow" if wandb_id is not None else 'never')
        wandb_id = wandb_run.id
        dist.print0(f"Wandb ID: {wandb_id}")

    # Dataset kwargs for Semantic Communication (RENEW_SC)
    sc_channel = 'slow_fading_eq' if opts.slow_fading_eq else opts.sc_channel_type
    c.dataset_kwargs = dnnlib.EasyDict(
        path=opts.data,
        renew_path=opts.data,
        adjscc_config=opts.adjscc_config,
        adjscc_model_path=opts.adjscc_model_path,
        image_dataset=opts.image_data,
        T=opts.sc_t,
        dataset_len=opts.sc_dataset_len,
        seed=opts.seed if opts.seed is not None else 42,
        matching_mode=opts.matching_mode,
        power_normalize_latent=True,
        view_as_complex=opts.view_as_complex,
        channel_type=sc_channel,
        slow_fading_eq=opts.slow_fading_eq,
        sigma=opts.sigma,
        additive_noise_sigma=opts.additive_noise_sigma,
        multiply_noise_sigma=opts.multiply_noise_sigma,
        only_additive_noise=opts.only_additive_noise,
        noise_mean_alter_way=opts.noise_mean_alter_way,
        noise_mean_flag=not opts.dynamic_noise_sigma,
        dynamic_noise_sigma=opts.dynamic_noise_sigma,
        corruption_probability_per_image=opts.corruption_probability,
        must_contain=opts.must_contain,
        must_not_contain=opts.must_not_contain,
        snr_range=opts.snr_range,
        normalize_value=opts.data_norm,
        use_labels=opts.cond,
        dataset_keep_percentage=opts.dataset_keep_percentage,
    )

    if opts.snr_range is not None:
        dist.print0(f'Using SNR range for natural noise corruption: {opts.snr_range} (additive_noise_sigma and multiply_noise_sigma are ignored)')

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=4 if opts.workers > 0 else None, persistent_workers=(opts.workers > 0))

    c.data_loader_kwargs.collate_fn = rf_augmentation_collate_fn(
        flip_probability=opts.flip_aug_ratio, 
        phase_shift_probability=opts.phase_shift_aug_ratio, 
        other_collate_fn=[torch.utils.data.default_collate, ],
        is_label_complex=opts.view_as_complex
    )

    # Determine whether to turn on validation
    if opts.validation_interval == -1:
        validation_on_off = False
    elif opts.validation_data is None and opts.dataset_keep_percentage >= 1.0:
        validation_on_off = False
    else:
        validation_on_off = True

    c.validation_kwargs = dnnlib.EasyDict(
        validation_on_off=validation_on_off, 
        validation_interval=opts.validation_interval, 
        validation_iterations=opts.validation_iterations, 
        validation_batch_size=opts.validation_batch_size, 
        validation_data=opts.validation_data
    )

    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8, weight_decay=opts.weight_decay)
    c.lr_rampup_kimg = opts.lr_rampup_kimg

    dist.synchronize()
    # Validate dataset options.
    try:
        dataset_obj = RenewChannelDataset(**c.dataset_kwargs)
        dist.print0(f"[RENEW_SC] Time steps T = {dataset_obj.T} (auto-calculated: {opts.sc_t is None})")
        dist.print0(f"[RENEW_SC] Channel type: {dataset_obj.channel_type} (equalization: {dataset_obj.channel_type == 'slow_fading_eq'})")
        dataset_name = copy.copy(dataset_obj.name)
        c.dataset_kwargs.dataset_keep_percentage = opts.dataset_keep_percentage
        c.dataset_kwargs.max_size = int(len(dataset_obj) * opts.dataset_keep_percentage)
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
        import gc
        gc.collect()
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,2], label_type='downlink' if opts.cond else 'no_label')
    elif opts.arch == 'ddpmpp_256':
        c.network_kwargs.update(model_type='RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=256, channel_mult=[1,2,2,2], label_type='downlink' if opts.cond else 'no_label')
    elif opts.arch == 'ddpmpp_192':
        c.network_kwargs.update(model_type='RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=192, channel_mult=[1,2,2,2], label_type='downlink' if opts.cond else 'no_label')
    elif opts.arch == 'ddpmpp_64':
        c.network_kwargs.update(model_type='RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=64, channel_mult=[1,2,2,2], label_type='downlink' if opts.cond else 'no_label')
    elif opts.arch == 'rf_dit':
        c.network_kwargs.update(
            model_type='RF_DiT',
            model_channels=64,
            channel_mult=[1, 2],
            num_blocks_per_stage=4,
            num_dit_blocks=5,
            dit_hidden_dim=256,
            dit_num_heads=8,
            dit_mlp_ratio=4.0,
            causal_conv=True,
            pos_embed_type='learnable_2d',
            resample_filter=[1, 1],
            label_type='downlink' if opts.cond else 'no_label',
        )
    elif opts.arch == 'rf_dit_large':
        c.network_kwargs.update(
            model_type='RF_DiT',
            model_channels=128,
            channel_mult=[1, 2],
            num_blocks_per_stage=4,
            num_dit_blocks=8,
            dit_hidden_dim=512,
            dit_num_heads=8,
            dit_mlp_ratio=4.0,
            causal_conv=True,
            pos_embed_type='learnable_2d',
            resample_filter=[1, 1],
            label_type='downlink' if opts.cond else 'no_label',
        )
    elif opts.arch == 'rf_dit_small':
        c.network_kwargs.update(
            model_type='RF_DiT',
            model_channels=32,
            channel_mult=[1, 2],
            num_blocks_per_stage=2,
            num_dit_blocks=3,
            dit_hidden_dim=128,
            dit_num_heads=4,
            dit_mlp_ratio=4.0,
            causal_conv=True,
            pos_embed_type='learnable_2d',
            resample_filter=[1, 1],
            label_type='downlink' if opts.cond else 'no_label',
        )
    elif opts.arch == 'rf_dit_real':
        c.network_kwargs.update(
            model_type='RF_DiT_Real',
            model_channels=64,
            channel_mult=[1, 2],
            num_blocks_per_stage=4,
            num_dit_blocks=5,
            dit_hidden_dim=256,
            dit_num_heads=8,
            dit_mlp_ratio=4.0,
            causal_conv=True,
            pos_embed_type='learnable_2d',
            resample_filter=[1, 1],
            label_type='downlink' if opts.cond else 'no_label',
        )
    elif opts.arch == 'rf_dit_real_large':
        c.network_kwargs.update(
            model_type='RF_DiT_Real',
            model_channels=128,
            channel_mult=[1, 2],
            num_blocks_per_stage=4,
            num_dit_blocks=9,
            dit_hidden_dim=512,
            dit_num_heads=8,
            dit_mlp_ratio=4.0,
            causal_conv=True,
            pos_embed_type='learnable_2d',
            resample_filter=[1, 1],
            label_type='downlink' if opts.cond else 'no_label',
        )
    elif opts.arch == 'rf_dit_real_deep':
        c.network_kwargs.update(
            model_type='RF_DiT_Real',
            model_channels=128,
            channel_mult=[1, 2],
            num_blocks_per_stage=4,
            num_dit_blocks=20,
            dit_hidden_dim=256,
            dit_num_heads=8,
            dit_mlp_ratio=4.0,
            causal_conv=True,
            pos_embed_type='learnable_2d',
            resample_filter=[1, 1],
            label_type='downlink' if opts.cond else 'no_label',
        )
    elif opts.arch == 'rf_dit_real_small':
        c.network_kwargs.update(
            model_type='RF_DiT_Real',
            model_channels=32,
            channel_mult=[1, 2],
            num_blocks_per_stage=2,
            num_dit_blocks=3,
            dit_hidden_dim=128,
            dit_num_heads=4,
            dit_mlp_ratio=4.0,
            causal_conv=True,
            pos_embed_type='learnable_2d',
            resample_filter=[1, 1],
            label_type='downlink' if opts.cond else 'no_label',
        )
    else:
        raise ValueError(f'Unsupported architecture for Semantic Communication: {opts.arch}')

    if not opts.not_dynamic_noise_sigma_model:
        if (opts.arch in ['ddpmpp', 'ddpmpp_256', 'ddpmpp_192', 'ddpmpp_64']):
            c.network_kwargs.update(dynamic_noise=True)
        elif opts.arch.startswith('rf_dit'):
            c.network_kwargs.update(dynamic_noise=False)
        else:
            raise ValueError(f'Dynamic noise sigma model is only supported for ddpmpp and rf_dit architectures for now, but got {opts.arch}')
    
    # Select batch size per GPU.
    batch_gpu = opts.batch_gpu
    batch_size = opts.batch
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    consistency_batch_size_per_gpu_total = opts.consistency_batch_size // dist.get_world_size()
    consistency_batch_size_per_gpu = consistency_batch_size_per_gpu_total // num_accumulation_rounds
    assert opts.consistency_batch_size == consistency_batch_size_per_gpu * dist.get_world_size() * num_accumulation_rounds

    if opts.real_precond == 'edm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    elif opts.real_precond == 'edm_c_skip':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond_c_skip'
    elif opts.real_precond == 'edm_input_scaling_test':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond_input_scaling_test'
        c.network_kwargs.sigma_input_scale = opts.sigma_input_scale
    elif opts.real_precond == 'edm_with_scheduler':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond_with_scheduler'
        c.network_kwargs.m = opts.m_dim
        c.network_kwargs.k = opts.k_top
        c.network_kwargs.rho = opts.rho
        c.network_kwargs.scheduler_mode = opts.scheduler_mode
        if opts.scheduler_mode in ['using_sigma_t_n', 'no_sigma_t_n', 'no_nn_scheduler_with_z', 'only_above_sigma_t_n']:
            c.network_kwargs.label_type = 'classes'
            c.network_kwargs.label_dim = opts.m_dim
        else:
            c.network_kwargs.label_type = 'no_label'
            c.network_kwargs.label_dim = 0
    else:
        assert False, f"Unsupported real_precond: {opts.real_precond}" 

    if opts.precond == 'edm_dynamic':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss_dynamic_sigma'
    elif opts.precond == 'edm':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
    elif opts.precond == 'edm_boosted_sigma':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss_boosted_sigma'
    elif opts.precond == 'edm_loss_scaling_test':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss_loss_scaling_test'
        c.loss_kwargs.sigma_loss_scaling = opts.sigma_loss_scaling
    elif opts.precond == 'edm_with_scheduler':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss_with_scheduler'
        c.loss_kwargs.kl_coeff = opts.kl_coeff
    elif opts.precond == 'edm_with_latent_z':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss_with_latent_z'
        c.loss_kwargs.kl_coeff = opts.kl_coeff
    else:
        assert False, f"Unsupported precond: {opts.precond}"
    
    c.loss_kwargs.update(consistency_batch_size_per_gpu=consistency_batch_size_per_gpu)
    # whether to use weight for the consistency terms
    c.loss_kwargs.update(with_weight=opts.with_weight)
    # whether to use gradient for the consistency terms
    c.loss_kwargs.update(with_grad=opts.with_grad)
    c.loss_kwargs.update(num_consistency_steps=opts.num_consistency_steps)
    c.loss_kwargs.update(num_primes=opts.num_primes)
    c.loss_kwargs.update(consistency_coeff=opts.consistency_coeff)
    c.loss_kwargs.update(no_asm=opts.no_asm)
    c.loss_kwargs.update(P_mean=opts.p_mean, P_std=opts.p_std, sigma_data=opts.sigma_data)

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.num_dit_blocks is not None:
        c.network_kwargs.num_dit_blocks = opts.num_dit_blocks
    if opts.dit_hidden_dim is not None:
        c.network_kwargs.dit_hidden_dim = opts.dit_hidden_dim
    if opts.dit_num_heads is not None:
        c.network_kwargs.dit_num_heads = opts.dit_num_heads
    if opts.dit_mlp_ratio is not None:
        c.network_kwargs.dit_mlp_ratio = opts.dit_mlp_ratio
    if opts.num_blocks_per_stage is not None:
        c.network_kwargs.num_blocks_per_stage = opts.num_blocks_per_stage
    if opts.causal_conv is not None:
        c.network_kwargs.causal_conv = opts.causal_conv
    if opts.pos_embed_type is not None:
        c.network_kwargs.pos_embed_type = opts.pos_embed_type
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16, label_dropout=opts.label_dropout, attention_scale=opts.attn_scale, sigma_data=opts.sigma_data)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    c.update(wandb_onoff=opts.wandb)
    c.update(task=opts.task)
    c.update(allow_tf32=opts.allow_tf32)
    c.update(grad_clip=opts.grad_clip)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\w+)\.pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        if match.group(1).isdecimal():
            c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}-{opts.wandb_group:s}-{opts.expr_id:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    elif resume_options is not None and 'run_dir' in resume_options:
        c.run_dir = resume_options['run_dir']
        dist.print0(f"Resuming training run in directory: {c.run_dir}")
    else:
        prev_run_dirs = []
        opts.outdir = os.path.join(opts.outdir, opts.wandb_group)
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        # add a random string of length 5 to run_dir
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}-{random_string}')
        assert not os.path.exists(c.run_dir)
    
    c.no_asm = opts.no_asm
    c.debug_test = opts.debug_test
    c.temp_save = opts.temp_save
    c.compile_network = opts.compile_network
    c.compile_mode = opts.compile_mode

    # Print options.
    if dist.get_rank() == 0:
        dist.print0()
        dist.print0('Training options:')
        # Create a deep copy for JSON serialization
        c_json = copy.deepcopy(c)
        c_json.wandb_id = wandb_id if opts.wandb else None
        # Convert function object to its name for serialization
        if 'collate_fn' in c_json.data_loader_kwargs and callable(c_json.data_loader_kwargs.collate_fn):
            c_json.data_loader_kwargs.collate_fn = c_json.data_loader_kwargs.collate_fn.__name__
        dist.print0(json.dumps(c_json, indent=2))
        dist.print0()
        dist.print0(f'Output directory:        {c.run_dir}')
        dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
        dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
        dist.print0(f'Network architecture:    {opts.arch}')
        if opts.arch.startswith('rf_dit'):
            dist.print0(f'  Model channels:        {c.network_kwargs.get("model_channels", 64)}')
            dist.print0(f'  Enc/Dec blocks/stage:  {c.network_kwargs.get("num_blocks_per_stage", 4)}')
            dist.print0(f'  DiT blocks:            {c.network_kwargs.get("num_dit_blocks", 5)}')
            dist.print0(f'  DiT hidden dim:        {c.network_kwargs.get("dit_hidden_dim", 256)}')
            dist.print0(f'  DiT num heads:         {c.network_kwargs.get("dit_num_heads", 8)}')
            dist.print0(f'  Causal conv on T:      {c.network_kwargs.get("causal_conv", True)}')
            dist.print0(f'  Positional embed:      {c.network_kwargs.get("pos_embed_type", "learnable_2d")}')
        dist.print0(f'Preconditioning & loss:  {opts.precond}')
        dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
        dist.print0(f'Batch size:              {c.batch_size}')
        dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
        dist.print0(f'torch.compile:           {opts.compile_network}' + (f' (mode={opts.compile_mode})' if opts.compile_network else ''))
        dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        dist.destroy_process_group()
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c_json, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

    dist.destroy_process_group()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
