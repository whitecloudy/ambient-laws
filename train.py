# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
from training.dataset import renewRfProcessedDataset, pad_collate_fn, rf_augmentation_collate_fn, WiDARDataset, XRF55Dataset
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
@click.option('--task',          help='Current task',         metavar='STR',                        type=str, required=True, show_default=True, default='RENEW')
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=str, default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm|edm_dynamic|edm_boosted_sigma|edm_loss_scaling_test',       type=click.Choice(['vp', 've', 'edm', 'edm_dynamic', 'edm_boosted_sigma', 'edm_loss_scaling_test']), default='edm', show_default=True)
@click.option('--real_precond',  help='Preconditioning & loss function', metavar='edm|edm_c_skip|edm_input_scaling_test',       type=click.Choice(['edm', 'edm_c_skip', 'edm_input_scaling_test']), default='edm', show_default=True)
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
@click.option('--complex_merge_axis', help='Axis to merge real and imaginary parts when view_as_complex is False. Set to "None" to not merge.', type=str, default='0', show_default=True)
@click.option('--transpose', help='Transpose the data axes according to the given order. Provide a list of two integers representing the new order of the first two axes (frame_resolution and ant_resolution). Set to None to not transpose.', type=str, default="0,1", show_default=True)
@click.option('--frame_res', help='Frame resolution of the RF data.', type=int, default=14, show_default=True)
@click.option('--ant_res', help='Antenna resolution of the RF data.', type=int, default=8, show_default=True)
@click.option('--data_norm', help='Data normalization value for the RF data.', type=float, default=1.0, show_default=True)
@click.option('--sigma_norm', help='Normalize data with sigma',                            is_flag=True)
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

# Validation params
@click.option('--validation_interval', help='How often to run validation. If -1, no validation will be run', metavar='tick', type=click.IntRange(min=-1), default=50, show_default=True)
@click.option('--validation_iterations', help='Number of iterations to run for validation.', metavar='INT', type=click.IntRange(min=1), default=5, show_default=True)
@click.option('--validation_batch_size', help='Batch size for validation.', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--validation_data', help='Path to the validation data. If not specified, portion of the training data will be used.', metavar='ZIP|DIR', type=str, default=None)

# Wandb related
@click.option('--wandb', help='Use wandb to log training progress',  type=bool, default=True)
@click.option('--wandb_group', help='Wandb group name',                   type=str, default='Test')
@click.option('--debug_test', help='Whether to run a quick test with 1 batch to verify the training loop works.', is_flag=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')

    complex_merge_axis = opts.complex_merge_axis
    if complex_merge_axis is None or str(complex_merge_axis).lower() == 'none':
        complex_merge_axis = None
    else:
        complex_merge_axis = int(complex_merge_axis)
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

    if dist.get_rank() == 0 and opts.wandb:
        # If there is resume_options is exist, and we are resuming from a certain checkpoint, then we will load and search for wandb ID. And if it exists, we will resume the wandb run only.
        wandb_id = None
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
        

    if opts.task == 'RENEW':
        # dataset_kwargs for RENEW dataset
        c.dataset_kwargs = dnnlib.EasyDict(path=opts.data, use_labels=opts.cond, cache=opts.cache, sigma=opts.sigma, 
                                        corruption_probability_per_image=opts.corruption_probability, corruption_probability_per_pixel=1.0, 
                                        only_positive=False, view_as_complex=opts.view_as_complex, complex_merge_axis=complex_merge_axis,
                                        resolution=(opts.frame_res, opts.ant_res), transpose=parse_int_list(opts.transpose) if opts.transpose is not None else None,
                                        normalize_value=opts.data_norm, must_contain=opts.must_contain, must_not_contain=opts.must_not_contain,
                                        multiply_noise_sigma=opts.multiply_noise_sigma, additive_noise_sigma=opts.additive_noise_sigma, only_additive_noise=opts.only_additive_noise, 
                                    noise_mean_alter_way=opts.noise_mean_alter_way,
                                    noise_mean_flag=not opts.dynamic_noise_sigma)
    elif opts.task == 'WIDAR':
        # dataset_kwargs for WIDAR dataset
        # TMP: 512 to 256 time scale for now to reduce the computational cost.
        c.dataset_kwargs = dnnlib.EasyDict(path=opts.data, use_labels=opts.cond, cache=opts.cache, sigma=opts.sigma, 
                                    corruption_probability_per_image=opts.corruption_probability, corruption_probability_per_pixel=1.0, 
                                    only_positive=False, view_as_complex=opts.view_as_complex, complex_merge_axis=complex_merge_axis,
                                    transpose=parse_int_list(opts.transpose) if opts.transpose is not None else None,
                                    normalize_value=opts.data_norm, must_contain=opts.must_contain, must_not_contain=opts.must_not_contain,
                                    multiply_noise_sigma=opts.multiply_noise_sigma, additive_noise_sigma=opts.additive_noise_sigma, only_additive_noise=opts.only_additive_noise, sigma_norm=opts.sigma_norm)
        # c.dataset_kwargs = dnnlib.EasyDict(path=opts.data, use_labels=opts.cond, cache=opts.cache, sigma=opts.sigma, 
        #                                    corruption_probability_per_image=opts.corruption_probability, corruption_probability_per_pixel=1.0, 
        #                                    only_positive=False, view_as_complex=opts.view_as_complex, complex_merge_axis=opts.complex_merge_axis,
        #                                    resolution=(3, 512, 30), transpose=parse_int_list(opts.transpose) if opts.transpose is not None else None,
        #                                    normalize_value=opts.data_norm, must_contain=opts.must_contain, must_not_contain=opts.must_not_contain,
        #                                    multiply_noise_sigma=opts.multiply_noise_sigma, additive_noise_sigma=opts.additive_noise_sigma, only_additive_noise=opts.only_additive_noise)
    elif opts.task == 'XRF55':
                c.dataset_kwargs = dnnlib.EasyDict(path=opts.data, use_labels=opts.cond, cache=opts.cache, sigma=opts.sigma, 
                                    corruption_probability_per_image=opts.corruption_probability, corruption_probability_per_pixel=1.0, 
                                    only_positive=False, view_as_complex=opts.view_as_complex, complex_merge_axis=complex_merge_axis,
                                    transpose=parse_int_list(opts.transpose) if opts.transpose is not None else None,
                                    normalize_value=opts.data_norm, must_contain=opts.must_contain, must_not_contain=opts.must_not_contain,
                                    multiply_noise_sigma=opts.multiply_noise_sigma, additive_noise_sigma=opts.additive_noise_sigma, only_additive_noise=opts.only_additive_noise, sigma_norm=opts.sigma_norm)


    else:
        raise ValueError(f'Unknown task: {opts.task}')
        
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=4, persistent_workers=True)

    is_label_complex = False
    if opts.task == 'RENEW':
        is_label_complex = True

    c.data_loader_kwargs.collate_fn = rf_augmentation_collate_fn(flip_probability=opts.flip_aug_ratio, 
                                                                phase_shift_probability=opts.phase_shift_aug_ratio, 
                                                                other_collate_fn=[torch.utils.data.default_collate, ],
                                                                is_label_complex=is_label_complex)
                                                                # other_collate_fn=[pad_collate_fn(dynamic_noise=opts.dynamic_noise_sigma), torch.utils.data.default_collate])
    validation_on_off = False
    # Determine whether to turn on validation
    if opts.validation_interval == -1:  # If validation_interval is -1, we will turn off validation regardless of other options.
        validation_on_off = False
    elif opts.task != 'XRF55' and opts.validation_data is None and opts.dataset_keep_percentage >= 1.0: # If validation_data is not specified and we are using the whole dataset for training, we will turn off validation.
        validation_on_off = False
    else:
        validation_on_off = True

    c.validation_kwargs = dnnlib.EasyDict(validation_on_off=validation_on_off, 
                                          validation_interval=opts.validation_interval, 
                                          validation_iterations=opts.validation_iterations, 
                                          validation_batch_size=opts.validation_batch_size, 
                                          validation_data=opts.validation_data)

    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8, weight_decay=opts.weight_decay)
    c.lr_rampup_kimg = opts.lr_rampup_kimg

    dist.synchronize()
    # Validate dataset options.
    try:
        if opts.task == 'RENEW':
            dataset_obj = renewRfProcessedDataset(**c.dataset_kwargs)
        elif opts.task == 'WIDAR':
            c.dataset_kwargs.must_contain = "regex:gesture(0|1|2|3|17|18)(?!\d)"
            dataset_obj = WiDARDataset(**c.dataset_kwargs)
        elif opts.task == 'XRF55':
            dataset_obj = XRF55Dataset(**c.dataset_kwargs)
        else:
            raise ValueError(f'Unknown task: {opts.task}')
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
    elif opts.arch == 'widar_ddpmpp':
        c.network_kwargs.update(model_type='WiDAR_RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[[1,1,1,1],[1,1]], resample_stride=[4,2], model_channels=16, channel_mult=[1,2,2,2], kernel_size = [9,3])
    elif opts.arch == 'widar_ddpmpp_stem256':
        c.network_kwargs.update(model_type='WiDAR_RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=256, channel_mult=[1,1,2,2], kernel_size=[3,3], stem_stride=[8,1], stem_kernel=[24,3], attn_resolutions=[16, 8])
    elif opts.arch == 'widar_ddpmpp_stem512':
        c.network_kwargs.update(model_type='WiDAR_RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=256, channel_mult=[1,2,2,2], kernel_size=[3,3], stem_stride=[16,1], stem_kernel=[48,3], attn_resolutions=[16, 8])
    elif opts.arch == 'ddpmpp_256':
        c.network_kwargs.update(model_type='RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=256, channel_mult=[1,2,2,2], label_type='downlink' if opts.cond else 'no_label')
    elif opts.arch == 'ddpmpp_192':
        c.network_kwargs.update(model_type='RF_SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=192, channel_mult=[1,2,2,2], label_type='downlink' if opts.cond else 'no_label')
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'rf_transformer_default':
        c.network_kwargs.update(model_type='RF_transformer', cond_dim=(6 if opts.cond else 0))
    elif opts.arch == 'rf_transformer_default_256':
        c.network_kwargs.update(model_type='RF_transformer', sample_rate=256, cond_dim=(6 if opts.cond else 0))
    elif opts.arch == 'rf_transformer_default_xrf_500':
        c.network_kwargs.update(model_type='RF_transformer', sample_rate=500, input_dim=270, cond_dim=55)
    elif opts.arch == 'rf_transformer_default_xrf_500_large':
        c.network_kwargs.update(model_type='RF_transformer', sample_rate=500, input_dim=270, cond_dim=55, hidden_dim=512, num_block=16)
    elif opts.arch == 'rf_transformer_MIMO_default':
        c.network_kwargs.update(model_type='RF_transformer_MIMO', sample_rate=1, input_dim=[8, 26])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    if not opts.not_dynamic_noise_sigma_model:
        if (opts.arch in ['ddpmpp', 'ddpmpp_256', 'ddpmpp_192']):
            c.network_kwargs.update(dynamic_noise=True)
        else:
            raise ValueError(f'Dynamic noise sigma model is only supported for ddpmpp architectures for now, but got {opts.arch}')
    
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
    else:
        assert False, f"Only edm, edm_c_skip, and edm_input_scaling_test are supported for now, but got {opts.real_precond}" 

    if opts.precond == 'edm_dynamic':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss_dynamic_sigma'
    elif opts.precond == 'edm':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
    elif opts.precond == 'edm_boosted_sigma':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss_boosted_sigma'
    elif opts.precond == 'edm_loss_scaling_test':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss_loss_scaling_test'
        c.loss_kwargs.sigma_loss_scaling = opts.sigma_loss_scaling
    else:
        assert False, f"Only edm, edm_dynamic, edm_boosted_sigma, and edm_loss_scaling_test are supported for now, but got {opts.precond}"
    
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
        print(c_json.data_loader_kwargs.collate_fn)
        dist.print0(json.dumps(c_json, indent=2))
        dist.print0()
        dist.print0(f'Output directory:        {c.run_dir}')
        dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
        dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
        dist.print0(f'Network architecture:    {opts.arch}')
        dist.print0(f'Preconditioning & loss:  {opts.precond}')
        dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
        dist.print0(f'Batch size:              {c.batch_size}')
        dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
        dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
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
