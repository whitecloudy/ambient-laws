# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
import joblib
from huggingface_hub import hf_hub_download
from training.dataset import renewRfDataset, renewRfProcessedDataset, widarRfDataset, pad_collate_fn, widar_collate_fn
import json
#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    stop_variance=0.0
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    x_list = []

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat.expand(x_hat.shape[0]), class_labels).to(torch.float64)
        x_list.append(denoised.clone().detach())
        
        # Stop if variance is below threshold
        if t_next ** 2 < stop_variance:
            return denoised

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next.expand(x_next.shape[0]), class_labels).to(torch.float64)
            x_list.append(denoised.clone().detach())
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        

    return x_next, x_list

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def truncated_edm_sampler(
    net, latents, class_labels=None, sigma=0.0, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    stop_variance=0.0
):
    batch_size = latents.shape[0]
    device = latents.device

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    # step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)

    step_indices = step_indices.expand(batch_size, -1)
    
    if sigma.ndim == 1:
        sigma = sigma.unsqueeze(1)
        
    sigma_min_rho = sigma ** (1 / rho)
    sigma_max_rho = sigma_max ** (1 / rho)
    
    # t_steps = (sigma_max_rho + step_indices / (num_steps - 1) * (sigma_min_rho - sigma_max_rho)) ** rho
    t_steps = (sigma_max_rho + step_indices / (num_steps) * (sigma_min_rho - sigma_max_rho)) ** rho

    # zeros = torch.zeros((t_steps.shape[0], 1), device=t_steps.device, dtype=t_steps.dtype)
    # t_steps = torch.cat([net.round_sigma(t_steps), zeros], dim=1) # t_N = 0
    print(t_steps)
    dim_diff = len(latents.shape) - (len(t_steps.shape) - 1)
    for _ in range(dim_diff):
        t_steps = t_steps.unsqueeze(-1)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0, 0]
    for i in range(num_steps): # 0, ..., N-1
        x_cur = x_next
        t_cur = t_steps[:, i]
        t_next = t_steps[:, i+1]

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur[0] <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        
        # Stop if variance is below threshold
        if torch.all(t_next ** 2 < stop_variance):
            return denoised

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    # temporary return empty list for debugging
    return x_next, []


#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next


def cal_SNR(predict : torch.Tensor, truth : torch.Tensor, complex_axis=None):
    if complex_axis != None:
        assert predict.dtype != torch.complex and truth.dtype != torch.complex, "\'complex axis\' is given while dtype is already complex!"
        # Handling batch axis
        if complex_axis >= 0:
            complex_axis += 1

        predict = torch.split(predict, 2, dim=complex_axis)
        truth = torch.split(truth, 2, dim=complex_axis)

        predict = predict[0] + 1j * predict[1]
        truth = truth[0] + 1j * truth[1]
    axis_list = [i for i in range(1, len(predict.shape))]
    PS = torch.sum(torch.abs(truth)**2, axis=axis_list)  # power of signal
    PN = torch.sum(torch.abs(predict - truth)**2, axis=axis_list)  # power of noise
    ratio = PS / PN
    return 10 * torch.log10(ratio)

class renew_with_average_image(renewRfProcessedDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        image_original = item['image']

        fname = str(item['filename'])

        # get other data from pilot pair
        if fname.find('pilot0') != -1:
            change_fname = fname.replace('pilot0', 'pilot1')
        elif fname.find('pilot1') != -1:
            change_fname = fname.replace('pilot1', 'pilot0')
        else:
            import warnings
            warnings.warn(f"Filename {fname} does not contain 'pilot0' or 'pilot1'. Returning original item.")
            return item

        image_pair = super().__create_item__(change_fname)['image']
        item['image'] = (image_original + image_pair) / 2.0
        
        return item
    
class renew_with_clear_image(renewRfProcessedDataset):
    def __init__(self, clear_label_dir, **kwargs):
        self.clear_label_dir = clear_label_dir
        super().__init__(**kwargs)

    def __create_item_without_noise_additive(self, fname):
        csi_data, noise_sigma_data = super()._load_and_normalize(fname)
        csi_data, noise_sigma_data = super()._apply_transpose(csi_data, noise_sigma_data)
        csi_data, noise_sigma_data = super()._handle_complex_view(csi_data, noise_sigma_data)

        return csi_data, noise_sigma_data


    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # Original filename would be frame<n1>_user<n2>_pilot<p>_cell<n3>_subcarrier<n4>.npz
        fname = str(item['filename'])
        
        # Clear filename would be frame<n1>_user<n2>_mean_cell<n3>_subcarrier<n4>.npz
        clear_fname = re.sub(r'pilot[01]', 'mean', fname)
        clear_fname = os.path.join(self.clear_label_dir, clear_fname)

        csi_data, noise_sigma_data = self.__create_item_without_noise_additive(clear_fname)
        csi_data, label_data = self._split_label(csi_data)
        item['image'] = csi_data
        
        return item



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


def load_hf_checkpoint(repo_id):
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    model_config = json.load(open(config_path, "r", encoding="utf-8"))
    model_config['class_name'] = 'training.networks.EDMPrecond'
    net = dnnlib.util.construct_class_by_name(**model_config)
    net = net.from_pretrained(repo_id)
    return net, model_config
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',   help='Network pickle filename', metavar='PATH|URL',                     type=str, required=True)
@click.option('--config_json',              help='Network config json filename', metavar='PATH|URL',                type=str, default=None, show_default=True)
@click.option('--seed',                     help='Random seed', metavar='INT',                                      type=int, default=11454, show_default=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--data',                     help='Path to the dataset', metavar='ZIP|DIR',                          type=str, required=True)
@click.option('--data_keep_ratio',          help='How much data keeping ratio', metavar='FLOAT',                    type=float, default=1.0, show_default=True)
@click.option('--flip_dataset',             help='Whether to flip the dataset to use the removed data',             is_flag=True)
@click.option('--must_contain',             help='Dataset name should contain', metavar='STR',                                 type=str, default=None, show_default=True)
@click.option('--must_not_contain',         help='Dataset name should not contain', metavar='STR',                           type=str, default=None, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',                help='Highest noise level  [default: varies]', metavar='FLOAT',         type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)
@click.option('--cond_additive_noise',      help='Whether to add noise to condition during training.',              type=float, default=0.0, show_default=True)
@click.option('--only_additive_noise',      help='Whether to only use additive noise for corruption without natural noise.', is_flag=True)

@click.option('--solver',                   help='Ablate ODE solver', metavar='euler|heun',                         type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',   help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm',type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                 help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',          type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                  help='Ablate signal scaling s(t)', metavar='vp|none',                   type=click.Choice(['vp', 'none']))
@click.option('--stop_variance',            help="Early stop generation at this variance",                          type=float, default=0.0)
@click.option('--trunc',                    help='Activate truncated sampling',                                     is_flag=True)


def main(network_pkl, config_json, subdirs, flip_dataset, seed, max_batch_size, data, data_keep_ratio, must_contain, must_not_contain, trunc, cond_additive_noise, only_additive_noise, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')

    if "pkl" in network_pkl:
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
        if config_json is not None:
            with open(config_json, "r", encoding="utf-8") as f:
                opts = json.load(f)
        else:
            config_filepath = os.path.join(os.path.dirname(network_pkl), 'training_options.json')
            assert os.path.isfile(config_filepath), f'Cannot find config file at {config_filepath}'
            dist.print0(f'Loading config from "{config_filepath}"...')
            with open(config_filepath, "r", encoding="utf-8") as f:
                opts = json.load(f)
    else:
        print("non pkl file is not supported yet.")
        exit(1)

    # opts = opts['dataset_kwargs']
    # dataset_kwargs for RENEW dataset
    dataset_kwargs = dnnlib.EasyDict(**opts['dataset_kwargs'])
    dataset_kwargs.path = data
    dataset_kwargs.dataset_keep_percentage = data_keep_ratio
    dataset_kwargs.must_contain = must_contain
    dataset_kwargs.must_not_contain = must_not_contain
    if flip_dataset:
        dataset_kwargs.flip_keep_dataset = True

    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=4, prefetch_factor=2)
    data_loader_kwargs.collate_fn = pad_collate_fn

    rnd_gen = torch.Generator(device=device).manual_seed(seed)

    dist.print0('Loading dataset...')
    # dataset_obj = renewRfProcessedDataset(**dataset_kwargs)
    # dataset_obj = renew_with_average_image(**dataset_kwargs)
    clear_label_dir, final_dir_name = os.path.split(data)
    clear_label_dir, _ = os.path.split(clear_label_dir)
    clear_label_dir = os.path.join(clear_label_dir, 'splited', final_dir_name)
    
    dataset_obj = renew_with_clear_image(clear_label_dir=clear_label_dir, **dataset_kwargs)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset_obj, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    dataloader_obj = torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dist_sampler, batch_size=max_batch_size, **data_loader_kwargs)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    total_SNR_sum = 0.0
    total_SNR_step_sum = torch.zeros(sampler_kwargs['num_steps']*2-1, dtype=torch.float64, device='cpu')

    # # Loop over batches.
    with torch.inference_mode(True):
        for dataset_item in tqdm.tqdm(dataloader_obj, unit='data', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            true_images = dataset_item["image"].to(device)                
            labels = dataset_item["label"].to(device)
            current_sigma = dataset_item["sigma"].to(device)
            if "original_shape" in dataset_item:
                original_shape = dataset_item["original_shape"].to(device)
            else:
                original_shape = None

            if only_additive_noise:
                current_sigma = torch.zeros_like(current_sigma)

            if cond_additive_noise > 0.0:
                noise = torch.randn_like(labels) * cond_additive_noise
                labels = labels + noise

            latents = torch.randn(true_images.shape, generator=rnd_gen, device=device)

            # # Pick latents and labels.
            # rnd = StackedRandomGenerator(device, batch_seeds)
            # latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            # class_labels = None
            # if net.label_dim:
            #     class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
            # if class_idx is not None:
            #     class_labels[:, :] = 0
            #     class_labels[:, class_idx] = 1

            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            # have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
            # sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
            sampler_fn = edm_sampler if not trunc else truncated_edm_sampler
            
            if sampler_fn == truncated_edm_sampler:
                gen_data, gen_data_each_list = sampler_fn(net, latents, labels, current_sigma, **sampler_kwargs)
            else:
                gen_data, gen_data_each_list = sampler_fn(net, latents, labels, **sampler_kwargs)

            if original_shape is not None:
                # Create a mask to zero out the loss on padded areas.
                # loss is expected to be of shape (N, C, H, W)
                mask = torch.zeros_like(gen_data)
                for i in range(gen_data.shape[0]):
                    # Get original shape for the i-th image
                    _, h, w = original_shape[i]
                    # Set mask to 1 for the original image area
                    mask[i, :, :h, :w] = 1
            else:
                mask = 1

            gen_data = gen_data * mask
            SNR = cal_SNR(gen_data, true_images, dataset_kwargs.complex_merge_axis)
            SNR_sum = torch.sum(SNR)

            SNR_step_sum = torch.zeros(len(gen_data_each_list), dtype=torch.float64, device=device)
            for i, gen_data_step in enumerate(gen_data_each_list):
                gen_data_step = gen_data_step * mask
                SNR_step = cal_SNR(gen_data_step, true_images, dataset_kwargs.complex_merge_axis)
                SNR_step_sum[i] = torch.sum(SNR_step)
            
            if dist.get_rank() == 0:
                collect_SNR_sum_list = [torch.tensor(0.0, dtype=torch.float64, device=device) for _ in range(dist.get_world_size())]
            else:
                collect_SNR_sum_list = None

            torch.distributed.gather(SNR_sum, gather_list=collect_SNR_sum_list, dst=0)

            if dist.get_rank() == 0:
                collect_SNR_step_sum_list = [torch.zeros_like(SNR_step_sum, dtype=torch.float64, device=device) for _ in range(dist.get_world_size())]
            else:
                collect_SNR_step_sum_list = None

            torch.distributed.gather(SNR_step_sum, gather_list=collect_SNR_step_sum_list, dst=0)

            if dist.get_rank() == 0:
                total_SNR_sum += sum(collect_SNR_sum_list).cpu().item()
                total_SNR_step_sum += sum(collect_SNR_step_sum_list).cpu()
            
            torch.distributed.barrier()
    dist.print0(f'Final SNR average : {total_SNR_sum / len(dataset_obj)}')
    dist.print0(f'Final SNR step average : {total_SNR_step_sum / len(dataset_obj)}')

    # Done.
    torch.distributed.barrier()

    dist.print0('Done.')
    dist.destroy_process_group()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
