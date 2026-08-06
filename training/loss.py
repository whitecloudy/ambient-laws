# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from numpy import pad
import numpy as np
import torch
from torch_utils import persistence
import ambient_utils
from training.sampler import edm_sampler, edm_sampler_with_scheduler, padding_mask_from_original_shape


# def from_x0_pred_to_xnature_pred_ve_to_ve_modify(x0_pred, noisy_input, current_sigma, desired_sigma):
def from_x0_pred_to_xnature_pred_ve_to_ve_modify(x0_pred, noisy_input, sigma_t, sigma_t_n):
    return (1 - (sigma_t_n / sigma_t) ** 2) * x0_pred + ((sigma_t_n / sigma_t) ** 2) * noisy_input

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, 
                 num_primes=4, num_consistency_steps=4, consistency_coeff=1.0, 
                 consistency_batch_size_per_gpu=4, with_weight=True, with_grad=False, no_asm=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.no_asm = no_asm

        self.num_consistency_steps = num_consistency_steps
        self.num_primes = num_primes
        self.consistency_coeff = consistency_coeff
        self.consistency_batch_size = consistency_batch_size_per_gpu
        self.with_weight = with_weight
        self.with_grad = with_grad

    def __call__(self, net, images, labels=None, current_sigma=0.0, augment_pipe=None, original_shape=None):
        current_sigma = torch.tensor(current_sigma)
        # net._set_static_graph()
        while current_sigma.ndim < images.ndim:
            current_sigma = current_sigma.unsqueeze(-1)
        current_sigma = current_sigma.expand_as(images)
        # current_sigma = current_sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        if original_shape is not None:
            padding_mask = padding_mask_from_original_shape(original_shape, images.shape)
        else:
            padding_mask = torch.ones_like(images)

        current_sigma = current_sigma * padding_mask
        
        rnd_normal = torch.randn([images.shape[0], ] + ([1] * (images.ndim - 1)), device=images.device)
        # sample a sigma in [current_sigma, sigma_T]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # min_sigma_of_each_batch = torch.amax(current_sigma, dim=tuple(range(1, current_sigma.ndim)), keepdim=True)
        # sigma = torch.clamp(sigma, min=min_sigma_of_each_batch + 1e-6)
        sigma = torch.clamp(sigma, min=current_sigma + 1e-6)
        y, augment_labels = (images, None)
        
        # add additional noise to reach the level sigma
        n = torch.randn_like(y) * torch.sqrt(sigma ** 2 - current_sigma ** 2)

        noisy_input = (y + n) * padding_mask
        x0_pred = net(noisy_input, sigma, labels, augment_labels=augment_labels)
        # make it xtn prediction

        # sigma가 0으로 남아있는걸 제거해서 nan 생성 방지
        if isinstance(sigma, torch.Tensor):
            nonzero_sigma = torch.where(sigma < 1e-8, torch.tensor(1e-8, dtype=sigma.dtype, device=sigma.device), sigma)
        elif sigma < 1e-8:
            nonzero_sigma = 1e-8
        else:
            nonzero_sigma = sigma

        # D_yn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, noisy_input, sigma, current_sigma)
        D_yn = from_x0_pred_to_xnature_pred_ve_to_ve_modify(x0_pred, noisy_input, nonzero_sigma, current_sigma)

        # loss weight depends on sigma
        weight = (nonzero_sigma ** 2 + self.sigma_data ** 2) / (nonzero_sigma * self.sigma_data) ** 2
        loss = weight * ((D_yn - y) ** 2)
        return_sigma = sigma
    
        # consistency loss
        if self.consistency_coeff > 0:
            # sample a new_sigma in [sigma, 0]
            new_rnd_normal = torch.randn_like(rnd_normal)
            new_sigma = (new_rnd_normal * self.P_std + self.P_mean).exp()
            new_sigma = torch.clamp(new_sigma, max=sigma)

            # we will only keep the first batch_size / self.num_primes part of the batch
            consistency_batch_size = self.consistency_batch_size
            if noisy_input.shape[0] < consistency_batch_size:
                consistency_batch_size = noisy_input.shape[0]
            noisy_input = noisy_input[:consistency_batch_size]
            sigma = sigma[:consistency_batch_size]
            new_sigma = new_sigma[:consistency_batch_size]
            if labels is not None:
                labels = labels[:consistency_batch_size]
            edm_padding_mask = padding_mask[:consistency_batch_size]

            # repeat everything num_primes times
            noisy_input = noisy_input.repeat_interleave(self.num_primes, dim=0)
            sigma = sigma.repeat_interleave(self.num_primes, dim=0)
            new_sigma = new_sigma.repeat_interleave(self.num_primes, dim=0)
            if labels is not None:
                labels = labels.repeat_interleave(self.num_primes, dim=0)
            edm_padding_mask = edm_padding_mask.repeat_interleave(self.num_primes, dim=0)

            # run sampler from sigma -> new_sigma
            with torch.no_grad() if not self.with_grad else torch.enable_grad():
                x_t_prime = edm_sampler(net, noisy_input, class_labels=labels, num_steps=self.num_consistency_steps, 
                                        sigma_min=new_sigma, sigma_max=sigma, padding_mask=edm_padding_mask) 
            # get predictions for x_t_prime
            x0_pred_prime = net(x_t_prime, new_sigma, labels)
            # group together predictions
            x0_pred_prime = x0_pred_prime.reshape(consistency_batch_size, self.num_primes, *x0_pred_prime.shape[1:])
            # average predictions
            average_x0_pred_prime = x0_pred_prime.mean(dim=1)
            # check difference to x0_pred
            consistency_loss = ((average_x0_pred_prime - x0_pred[:consistency_batch_size]) ** 2)
            consistency_weight = weight[:consistency_batch_size] if self.with_weight else 1.0
            # loss[:consistency_batch_size] += self.consistency_coeff * consistency_weight * consistency_loss
            
            # In-place 연산 방지
            loss_consistency_part = self.consistency_coeff * consistency_weight * consistency_loss
            loss = torch.cat([loss[:consistency_batch_size] + loss_consistency_part, loss[consistency_batch_size:]], dim=0)
        
        loss = loss * padding_mask
        return loss, x0_pred, return_sigma, None

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).
@persistence.persistent_class
class EDMLoss_with_scheduler:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, 
                 num_primes=4, num_consistency_steps=4, consistency_coeff=1.0, 
                 consistency_batch_size_per_gpu=4, with_weight=True, with_grad=False, no_asm=False,
                 sigma_max=80, kl_coeff=1.0, **kwargs):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.no_asm = no_asm

        self.num_consistency_steps = num_consistency_steps
        self.num_primes = num_primes
        self.consistency_coeff = consistency_coeff
        self.consistency_batch_size = consistency_batch_size_per_gpu
        self.with_weight = with_weight
        self.with_grad = with_grad
        self.sigma_max = sigma_max
        self.kl_coeff = kl_coeff

    def __call__(self, net, images, labels=None, current_sigma=0.0, augment_pipe=None, original_shape=None):
        images_f64 = images.to(torch.float64)
        current_sigma_f64 = torch.as_tensor(current_sigma, dtype=torch.float64, device=images.device)
        # net._set_static_graph()
        while current_sigma_f64.ndim < images_f64.ndim:
            current_sigma_f64 = current_sigma_f64.unsqueeze(-1)
        current_sigma_f64 = current_sigma_f64.expand_as(images_f64)

        if original_shape is not None:
            padding_mask = padding_mask_from_original_shape(original_shape, images.shape)
        else:
            padding_mask = torch.ones_like(images)

        rnd_normal = torch.randn([images.shape[0], ] + ([1] * (images.ndim - 1)), device=images.device, dtype=torch.float64)
        # sample a sigma in reference space
        sigma_ref_f64 = (rnd_normal * self.P_std + self.P_mean).exp()

        sigma_t_n_mean = torch.sqrt(torch.mean(current_sigma_f64 ** 2, dim=list(range(1, current_sigma_f64.ndim))))
        sigma_t_n_mean_aligned = sigma_t_n_mean.reshape(-1, *[1] * (current_sigma_f64.ndim - 1))
        
        # Clamp sigma_ref to be at least sigma_t_n_mean to ensure we only add noise
        sigma_ref_f64 = torch.clamp(sigma_ref_f64, min=sigma_t_n_mean_aligned + 1e-6)

        def _get_net_attr(n, name):
            if hasattr(n, name):
                return getattr(n, name)
            elif hasattr(n, 'module') and hasattr(n.module, name):
                return getattr(n.module, name)
            return None

        generate_latent_z_fn = _get_net_attr(net, "generate_latent_z")
        get_noise_scheduling_fn = _get_net_attr(net, "get_noise_scheduling")
        assert generate_latent_z_fn is not None and get_noise_scheduling_fn is not None, "net must have generate_latent_z and get_noise_scheduling methods in EDMLoss_with_scheduler"

        z, kl_loss = generate_latent_z_fn(images_f64, current_sigma_f64)
        noise_schedule_dict = get_noise_scheduling_fn(sigma_ref_f64, current_sigma_f64, z)
        
        # get sigma in data space (float64): 
        sigma_f64 = noise_schedule_dict['poly_sigma'].to(torch.float64)
        abd = noise_schedule_dict['abd']

        y_f64, augment_labels = (images_f64, None)
        
        # add additional noise to reach the level sigma
        diff_sq = torch.clamp(sigma_f64 ** 2 - current_sigma_f64 ** 2, min=1e-12)
        n_f64 = torch.randn_like(y_f64, dtype=torch.float64) * torch.sqrt(diff_sq)

        noisy_input_f64 = (y_f64 + n_f64) * padding_mask
        is_classes = (hasattr(net, 'model') and getattr(net.model, 'label_type', None) == 'classes') or (hasattr(net, 'module') and hasattr(net.module, 'model') and getattr(net.module.model, 'label_type', None) == 'classes')
        class_labels_to_pass = z if (labels is None or is_classes or (isinstance(labels, torch.Tensor) and labels.ndim >= 2 and labels.shape[-1] == 0)) else labels
        x0_pred = net(noisy_input_f64, sigma_f64, class_labels=class_labels_to_pass, augment_labels=augment_labels, z=z).to(torch.float64)
        # make it xtn prediction

        # sigma가 0으로 남아있는걸 제거해서 nan 생성 방지
        if isinstance(sigma_f64, torch.Tensor):
            nonzero_sigma_f64 = torch.where(sigma_f64 == 0.0, torch.tensor(1e-8, dtype=torch.float64, device=sigma_f64.device), sigma_f64)
        elif sigma_f64 == 0.0:
            nonzero_sigma_f64 = 1e-8
        else:
            nonzero_sigma_f64 = sigma_f64

        D_yn_f64 = from_x0_pred_to_xnature_pred_ve_to_ve_modify(x0_pred, noisy_input_f64, nonzero_sigma_f64, current_sigma_f64)

        # loss weight depends on sigma (float64)
        weight_f64 = (nonzero_sigma_f64 ** 2 + self.sigma_data ** 2) / (nonzero_sigma_f64 * self.sigma_data) ** 2
        loss = weight_f64 * ((D_yn_f64 - y_f64) ** 2) + self.kl_coeff * kl_loss.to(torch.float64)
        return_sigma = sigma_f64
    
        # consistency loss
        if self.consistency_coeff > 0:
            # sample a new_sigma in reference space
            new_rnd_normal = torch.randn_like(rnd_normal, dtype=torch.float64)
            new_sigma_ref_f64 = (new_rnd_normal * self.P_std + self.P_mean).exp()
            new_sigma_ref_f64 = torch.clamp(new_sigma_ref_f64, max=sigma_ref_f64)
            
            # Map new_sigma_ref to multivariable space using scheduler
            new_noise_schedule_dict = get_noise_scheduling_fn(new_sigma_ref_f64, current_sigma_f64, z=z, abd=abd)
            new_sigma_f64 = new_noise_schedule_dict['poly_sigma'].to(torch.float64)

            # we will only keep the first batch_size / self.num_primes part of the batch
            consistency_batch_size = self.consistency_batch_size
            if noisy_input_f64.shape[0] < consistency_batch_size:
                consistency_batch_size = noisy_input_f64.shape[0]
            c_noisy_input_f64 = noisy_input_f64[:consistency_batch_size]
            c_sigma_f64 = sigma_f64[:consistency_batch_size]
            c_new_sigma_f64 = new_sigma_f64[:consistency_batch_size]
            c_sigma_ref_f64 = sigma_ref_f64[:consistency_batch_size]
            c_new_sigma_ref_f64 = new_sigma_ref_f64[:consistency_batch_size]
            c_current_sigma_f64 = current_sigma_f64[:consistency_batch_size]
            c_z = z[:consistency_batch_size]
            c_abd = (abd[0][:consistency_batch_size], abd[1][:consistency_batch_size], abd[2][:consistency_batch_size]) if abd is not None else None
            if labels is not None:
                c_labels_orig = labels[:consistency_batch_size]
            else:
                c_labels_orig = None
            c_edm_padding_mask = padding_mask[:consistency_batch_size]

            # repeat interleave logic for consistency loss
            c_noisy_input_f64 = c_noisy_input_f64.repeat_interleave(self.num_primes, dim=0)
            c_sigma_f64 = c_sigma_f64.repeat_interleave(self.num_primes, dim=0)
            c_new_sigma_f64 = c_new_sigma_f64.repeat_interleave(self.num_primes, dim=0)
            c_sigma_ref_f64 = c_sigma_ref_f64.repeat_interleave(self.num_primes, dim=0)
            c_new_sigma_ref_f64 = c_new_sigma_ref_f64.repeat_interleave(self.num_primes, dim=0)
            c_current_sigma_f64 = c_current_sigma_f64.repeat_interleave(self.num_primes, dim=0)
            c_z = c_z.repeat_interleave(self.num_primes, dim=0)
            if c_abd is not None:
                c_abd = (c_abd[0].repeat_interleave(self.num_primes, dim=0), c_abd[1].repeat_interleave(self.num_primes, dim=0), c_abd[2].repeat_interleave(self.num_primes, dim=0))
            if c_labels_orig is not None:
                c_labels_orig = c_labels_orig.repeat_interleave(self.num_primes, dim=0)
            c_edm_padding_mask = c_edm_padding_mask.repeat_interleave(self.num_primes, dim=0)

            consistency_noise_schedule_dict = {
                'poly_sigma': c_sigma_f64,
                'latent_z': c_z,
                'abd': c_abd,
                'sigma_ref': c_sigma_ref_f64,
                'current_sigma': c_current_sigma_f64,
            }

            c_class_labels_to_pass = c_z if (c_labels_orig is None or is_classes or (isinstance(c_labels_orig, torch.Tensor) and c_labels_orig.ndim >= 2 and c_labels_orig.shape[-1] == 0)) else c_labels_orig

            # run sampler from sigma -> new_sigma
            with torch.no_grad() if not self.with_grad else torch.enable_grad():
                x_t_prime = edm_sampler_with_scheduler(
                    net, c_noisy_input_f64, class_labels=c_class_labels_to_pass, num_steps=self.num_consistency_steps, 
                    sigma_min=c_new_sigma_f64, sigma_max=c_sigma_f64, padding_mask=c_edm_padding_mask,
                    noise_schedule_dict=consistency_noise_schedule_dict, new_sigma_ref=c_new_sigma_ref_f64
                ).to(torch.float64)
            # get predictions for x_t_prime
            x0_pred_prime = net(x_t_prime, c_new_sigma_f64, class_labels=c_class_labels_to_pass, z=c_z).to(torch.float64)
            # group together predictions
            x0_pred_prime = x0_pred_prime.reshape(consistency_batch_size, self.num_primes, *x0_pred_prime.shape[1:])
            # average predictions
            average_x0_pred_prime = x0_pred_prime.mean(dim=1)
            # check difference to x0_pred
            consistency_loss = ((average_x0_pred_prime - x0_pred[:consistency_batch_size]) ** 2)
            consistency_weight = weight_f64[:consistency_batch_size] if self.with_weight else 1.0
            
            loss_consistency_part = self.consistency_coeff * consistency_weight * consistency_loss
            loss = torch.cat([loss[:consistency_batch_size] + loss_consistency_part, loss[consistency_batch_size:]], dim=0)

        loss = loss * padding_mask
        return loss, x0_pred, return_sigma, kl_loss


#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss_loss_scaling_test:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, 
                 num_primes=4, num_consistency_steps=4, consistency_coeff=1.0, 
                 consistency_batch_size_per_gpu=4, with_weight=True, with_grad=False, no_asm=False, sigma_loss_scaling=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_loss_scaling = sigma_loss_scaling

        self.no_asm = no_asm

        self.num_consistency_steps = num_consistency_steps
        self.num_primes = num_primes
        self.consistency_coeff = consistency_coeff
        self.consistency_batch_size = consistency_batch_size_per_gpu
        self.with_weight = with_weight
        self.with_grad = with_grad

    def __call__(self, net, images, labels=None, current_sigma=0.0, augment_pipe=None, original_shape=None):
        current_sigma = torch.tensor(current_sigma)
        # net._set_static_graph()
        while current_sigma.ndim < images.ndim:
            current_sigma = current_sigma.unsqueeze(-1)
        current_sigma = current_sigma.expand_as(images)
        # current_sigma = current_sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        if original_shape is not None:
            padding_mask = padding_mask_from_original_shape(original_shape, images.shape)
        else:
            padding_mask = torch.ones_like(images)

        current_sigma = current_sigma * padding_mask
        
        rnd_normal = torch.randn([images.shape[0], ] + ([1] * (images.ndim - 1)), device=images.device)
        # sample a sigma in [current_sigma, sigma_T]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # TEMP : temporary change for check the impact of sigma max clamp during varying ASM Loss
        min_sigma_of_each_batch = torch.amax(current_sigma, dim=tuple(range(1, current_sigma.ndim)), keepdim=True)
        sigma = torch.clamp(sigma, min=min_sigma_of_each_batch + 1e-6)
        # sigma = torch.clamp(sigma, min=current_sigma + 1e-6)
        y, augment_labels = (images, None)
        
        # add additional noise to reach the level sigma
        n = torch.randn_like(y) * torch.sqrt(sigma ** 2 - current_sigma ** 2)

        noisy_input = (y + n) * padding_mask
        x0_pred = net(noisy_input, sigma, labels, augment_labels=augment_labels)
        # make it xtn prediction

        # sigma가 0으로 남아있는걸 제거해서 nan 생성 방지
        if isinstance(sigma, torch.Tensor):
            nonzero_sigma = torch.where(sigma < 1e-8, torch.tensor(1e-8, dtype=sigma.dtype, device=sigma.device), sigma)
        elif sigma < 1e-8:
            nonzero_sigma = 1e-8
        else:
            nonzero_sigma = sigma

        # D_yn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, noisy_input, sigma, current_sigma)
        D_yn = from_x0_pred_to_xnature_pred_ve_to_ve_modify(x0_pred, noisy_input, nonzero_sigma, current_sigma)

        # loss weight depends on sigma
        weight = (nonzero_sigma ** 2 + self.sigma_data ** 2) / (nonzero_sigma * self.sigma_data) ** 2

        # TEST: loss scaling weight
        loss_scaling_weight = (self.sigma_data**2 * sigma**2 + self.sigma_loss_scaling**4)/(self.sigma_loss_scaling**2 * (self.sigma_loss_scaling**2 + sigma**2))

        loss = weight * loss_scaling_weight * ((D_yn - y) ** 2)
        return_sigma = sigma
    
        # consistency loss
        if self.consistency_coeff > 0:
            # sample a new_sigma in [sigma, 0]
            new_rnd_normal = torch.randn_like(rnd_normal)
            new_sigma = (new_rnd_normal * self.P_std + self.P_mean).exp()
            new_sigma = torch.clamp(new_sigma, max=sigma)

            # we will only keep the first batch_size / self.num_primes part of the batch
            consistency_batch_size = self.consistency_batch_size
            if noisy_input.shape[0] < consistency_batch_size:
                consistency_batch_size = noisy_input.shape[0]
            noisy_input = noisy_input[:consistency_batch_size]
            sigma = sigma[:consistency_batch_size]
            new_sigma = new_sigma[:consistency_batch_size]
            if labels is not None:
                labels = labels[:consistency_batch_size]
            edm_padding_mask = padding_mask[:consistency_batch_size]

            # repeat everything num_primes times
            noisy_input = noisy_input.repeat_interleave(self.num_primes, dim=0)
            sigma = sigma.repeat_interleave(self.num_primes, dim=0)
            new_sigma = new_sigma.repeat_interleave(self.num_primes, dim=0)
            if labels is not None:
                labels = labels.repeat_interleave(self.num_primes, dim=0)
            edm_padding_mask = edm_padding_mask.repeat_interleave(self.num_primes, dim=0)

            # run sampler from sigma -> new_sigma
            with torch.no_grad() if not self.with_grad else torch.enable_grad():
                x_t_prime = edm_sampler(net, noisy_input, class_labels=labels, num_steps=self.num_consistency_steps, 
                                        sigma_min=new_sigma, sigma_max=sigma, padding_mask=edm_padding_mask) 
            # get predictions for x_t_prime
            x0_pred_prime = net(x_t_prime, new_sigma, labels)
            # group together predictions
            x0_pred_prime = x0_pred_prime.reshape(consistency_batch_size, self.num_primes, *x0_pred_prime.shape[1:])
            # average predictions
            average_x0_pred_prime = x0_pred_prime.mean(dim=1)
            # check difference to x0_pred
            consistency_loss = ((average_x0_pred_prime - x0_pred[:consistency_batch_size]) ** 2)
            consistency_weight = weight[:consistency_batch_size] if self.with_weight else 1.0
            # loss[:consistency_batch_size] += self.consistency_coeff * consistency_weight * consistency_loss
            
            # In-place 연산 방지
            loss_consistency_part = self.consistency_coeff * consistency_weight * consistency_loss
            loss = torch.cat([loss[:consistency_batch_size] + loss_consistency_part, loss[consistency_batch_size:]], dim=0)
        
        loss = loss * padding_mask
        return loss, x0_pred, return_sigma, None

#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
# EDM Loss with dynamic sigma in log normal distribution
# Default lognorm dist parameter is calculated from RENEW dataset "ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS"
@persistence.persistent_class
class EDMLoss_dynamic_sigma:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, 
                 num_primes=4, num_consistency_steps=4, consistency_coeff=1.0, 
                 consistency_batch_size_per_gpu=4, with_weight=True, with_grad=False, no_asm=False,
                 lognorm_dist_mean = 0.11322956881040724,
                 lognorm_dist_loc = -0.3077225803161989,
                 lognorm_dist_sigma = 0.3644639849662781):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        self.num_consistency_steps = num_consistency_steps
        self.num_primes = num_primes
        self.consistency_coeff = consistency_coeff
        self.consistency_batch_size = consistency_batch_size_per_gpu
        self.with_weight = with_weight
        self.with_grad = with_grad
        self.no_asm = no_asm

        self.lognorm_dist_mean = lognorm_dist_mean
        self.lognorm_dist_loc = lognorm_dist_loc
        self.lognorm_dist_sigma = lognorm_dist_sigma

    def __get_lognormal_values(self, size):
        return np.random.lognormal(self.lognorm_dist_mean, self.lognorm_dist_sigma, size) + self.lognorm_dist_loc

    def __call__(self, net, images, labels=None, current_sigma=0.0, augment_pipe=None, original_shape=None):
        current_sigma = torch.tensor(current_sigma)
        # net._set_static_graph()
        while current_sigma.ndim < images.ndim:
            current_sigma = current_sigma.unsqueeze(-1)
        current_sigma = current_sigma.expand_as(images)
        # current_sigma = current_sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        if original_shape is not None:
            padding_mask = padding_mask_from_original_shape(original_shape, images.shape)
        else:
            padding_mask = 1
        
        current_sigma = current_sigma * padding_mask
        
        rnd_normal = torch.randn([images.shape[0], ] + ([1] * (images.ndim - 1)), device=images.device)
        # sample a sigma in [current_sigma, sigma_T]
        sigma_center = (rnd_normal * self.P_std + self.P_mean).exp()        
        sigma = torch.tensor(self.__get_lognormal_values(images.shape), device=images.device, dtype=torch.float32) * sigma_center
        sigma = torch.clamp(sigma, min=current_sigma + 1e-5) * padding_mask

        y, augment_labels = (images, None)

        # add additional noise to reach the level sigma
        n = torch.randn_like(y) * torch.sqrt(sigma ** 2 - current_sigma ** 2)

        noisy_input = (y + n) * padding_mask
        x0_pred = net(noisy_input, sigma, labels, augment_labels=augment_labels)
        # make it xtn prediction

        # sigma가 0으로 남아있는걸 제거해서 nan 생성 방지
        if isinstance(sigma, torch.Tensor):
            nonzero_sigma = torch.where(sigma == 0.0, torch.tensor(1e-8, dtype=sigma.dtype, device=sigma.device), sigma)
        elif sigma == 0.0:
            nonzero_sigma = 1e-8
        else:
            nonzero_sigma = sigma

        # D_yn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, noisy_input, sigma, current_sigma)
        D_yn = from_x0_pred_to_xnature_pred_ve_to_ve_modify(x0_pred, noisy_input, nonzero_sigma, current_sigma)
        # loss weight depends on sigma
        weight = (nonzero_sigma ** 2 + self.sigma_data ** 2) / (nonzero_sigma * self.sigma_data) ** 2
        loss = weight * ((D_yn - y) ** 2)
        return_sigma = sigma
    
        # consistency loss
        if self.consistency_coeff > 0:
            # sample a new_sigma in [sigma, 0]
            new_rnd_normal = torch.randn_like(rnd_normal)
            new_sigma_center = (new_rnd_normal * self.P_std + self.P_mean).exp()
            new_sigma = torch.tensor(self.__get_lognormal_values(images.shape), device=images.device, dtype=torch.float32) * new_sigma_center

            new_sigma = torch.clamp(new_sigma, max=nonzero_sigma)
            new_sigma = torch.clamp(new_sigma, min=1e-8)

            # we will only keep the first batch_size / self.num_primes part of the batch
            consistency_batch_size = self.consistency_batch_size
            noisy_input = noisy_input[:consistency_batch_size]
            sigma_for_sampler = nonzero_sigma[:consistency_batch_size]
            new_sigma = new_sigma[:consistency_batch_size]
            if labels is not None:
                labels = labels[:consistency_batch_size]
            if isinstance(padding_mask, torch.Tensor):
                edm_padding_mask = padding_mask[:consistency_batch_size]
            else:
                edm_padding_mask = padding_mask

            # repeat everything num_primes times
            noisy_input = noisy_input.repeat_interleave(self.num_primes, dim=0)
            sigma_for_sampler = sigma_for_sampler.repeat_interleave(self.num_primes, dim=0)
            new_sigma = new_sigma.repeat_interleave(self.num_primes, dim=0)
            if labels is not None:
                labels = labels.repeat_interleave(self.num_primes, dim=0)
            if isinstance(edm_padding_mask, torch.Tensor):
                edm_padding_mask = edm_padding_mask.repeat_interleave(self.num_primes, dim=0)

            # run sampler from sigma -> new_sigma
            with torch.no_grad() if not self.with_grad else torch.enable_grad():
                x_t_prime = edm_sampler(net, noisy_input, class_labels=labels, num_steps=self.num_consistency_steps, 
                                        sigma_min=new_sigma, sigma_max=sigma_for_sampler, padding_mask=edm_padding_mask) 
            # get predictions for x_t_prime
            x0_pred_prime = net(x_t_prime, new_sigma, labels)
            # group together predictions
            x0_pred_prime = x0_pred_prime.reshape(consistency_batch_size, self.num_primes, *x0_pred_prime.shape[1:])
            # average predictions
            average_x0_pred_prime = x0_pred_prime.mean(dim=1)
            # check difference to x0_pred
            consistency_loss = ((average_x0_pred_prime - x0_pred[:consistency_batch_size]) ** 2)
            consistency_weight = weight[:consistency_batch_size] if self.with_weight else 1.0
            # loss[:consistency_batch_size] += self.consistency_coeff * consistency_weight * consistency_loss
            
            # In-place 연산 방지
            loss_consistency_part = self.consistency_coeff * consistency_weight * consistency_loss
            loss = torch.cat([loss[:consistency_batch_size] + loss_consistency_part, loss[consistency_batch_size:]], dim=0)
        
        loss = loss * padding_mask
        return loss, x0_pred, return_sigma, None

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# EDM Loss with dynamic sigma in log normal distribution
# Default lognorm dist parameter is calculated from RENEW dataset "ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS"
@persistence.persistent_class
class EDMLoss_boosted_sigma:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, 
                 num_primes=4, num_consistency_steps=4, consistency_coeff=1.0, 
                 consistency_batch_size_per_gpu=4, with_weight=True, with_grad=False, no_asm=False):
                 
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        self.num_consistency_steps = num_consistency_steps
        self.num_primes = num_primes
        self.consistency_coeff = consistency_coeff
        self.consistency_batch_size = consistency_batch_size_per_gpu
        self.with_weight = with_weight
        self.with_grad = with_grad
        self.no_asm = no_asm

    def __call__(self, net, images, labels=None, current_sigma=0.0, augment_pipe=None, original_shape=None):
        current_sigma = torch.tensor(current_sigma)
        # net._set_static_graph()
        while current_sigma.ndim < images.ndim:
            current_sigma = current_sigma.unsqueeze(-1)
        current_sigma = current_sigma.expand_as(images)
        # current_sigma = current_sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        if original_shape is not None:
            padding_mask = padding_mask_from_original_shape(original_shape, images.shape)
        else:
            padding_mask = 1
        
        current_sigma = current_sigma * padding_mask

        normalized_current_sigma = (current_sigma / (torch.mean(current_sigma**2, dim=tuple(range(1, current_sigma.ndim)), keepdim=True)**0.5)).to(torch.float64).to(images.device)

        if self.no_asm:
            current_sigma = 0.0
        
        rnd_normal = torch.randn([images.shape[0], ] + ([1] * (images.ndim - 1)), device=images.device)
        # sample a sigma in [current_sigma, sigma_T]
        sigma_center = (rnd_normal * self.P_std + self.P_mean).exp()        
        sigma = normalized_current_sigma * sigma_center
        sigma = torch.clamp(sigma, min=current_sigma + 1e-5) * padding_mask

        y, augment_labels = (images, None)

        # add additional noise to reach the level sigma
        n = torch.randn_like(y) * torch.sqrt(sigma ** 2 - current_sigma ** 2)

        noisy_input = (y + n) * padding_mask
        x0_pred = net(noisy_input, sigma, labels, augment_labels=augment_labels)
        # make it xtn prediction

        # sigma가 0으로 남아있는걸 제거해서 nan 생성 방지
        if isinstance(sigma, torch.Tensor):
            nonzero_sigma = torch.where(sigma == 0.0, torch.tensor(1e-8, dtype=sigma.dtype, device=sigma.device), sigma)
        elif sigma == 0.0:
            nonzero_sigma = 1e-8
        else:
            nonzero_sigma = sigma

        # D_yn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, noisy_input, sigma, current_sigma)
        D_yn = from_x0_pred_to_xnature_pred_ve_to_ve_modify(x0_pred, noisy_input, nonzero_sigma, current_sigma)
        # loss weight depends on sigma
        weight = (nonzero_sigma ** 2 + self.sigma_data ** 2) / (nonzero_sigma * self.sigma_data) ** 2
        loss = weight * ((D_yn - y) ** 2)
        return_sigma = sigma
    
        # consistency loss
        if self.consistency_coeff > 0:
            # sample a new_sigma in [sigma, 0]
            new_rnd_normal = torch.randn_like(rnd_normal)
            new_sigma_center = (new_rnd_normal * self.P_std + self.P_mean).exp()
            new_sigma = normalized_current_sigma * new_sigma_center

            new_sigma = torch.clamp(new_sigma, max=nonzero_sigma)
            new_sigma = torch.clamp(new_sigma, min=1e-8)

            # we will only keep the first batch_size / self.num_primes part of the batch
            consistency_batch_size = self.consistency_batch_size
            noisy_input = noisy_input[:consistency_batch_size]
            sigma_for_sampler = nonzero_sigma[:consistency_batch_size]
            new_sigma = new_sigma[:consistency_batch_size]
            if labels is not None:
                labels = labels[:consistency_batch_size]
            if isinstance(padding_mask, torch.Tensor):
                edm_padding_mask = padding_mask[:consistency_batch_size]
            else:
                edm_padding_mask = padding_mask

            # repeat everything num_primes times
            noisy_input = noisy_input.repeat_interleave(self.num_primes, dim=0)
            sigma_for_sampler = sigma_for_sampler.repeat_interleave(self.num_primes, dim=0)
            new_sigma = new_sigma.repeat_interleave(self.num_primes, dim=0)
            if labels is not None:
                labels = labels.repeat_interleave(self.num_primes, dim=0)
            if isinstance(edm_padding_mask, torch.Tensor):
                edm_padding_mask = edm_padding_mask.repeat_interleave(self.num_primes, dim=0)

            # run sampler from sigma -> new_sigma
            with torch.no_grad() if not self.with_grad else torch.enable_grad():
                x_t_prime = edm_sampler(net, noisy_input, class_labels=labels, num_steps=self.num_consistency_steps, 
                                        sigma_min=new_sigma, sigma_max=sigma_for_sampler, padding_mask=edm_padding_mask) 
            # get predictions for x_t_prime
            x0_pred_prime = net(x_t_prime, new_sigma, labels)
            # group together predictions
            x0_pred_prime = x0_pred_prime.reshape(consistency_batch_size, self.num_primes, *x0_pred_prime.shape[1:])
            # average predictions
            average_x0_pred_prime = x0_pred_prime.mean(dim=1)
            # check difference to x0_pred
            consistency_loss = ((average_x0_pred_prime - x0_pred[:consistency_batch_size]) ** 2)
            consistency_weight = weight[:consistency_batch_size] if self.with_weight else 1.0
            # loss[:consistency_batch_size] += self.consistency_coeff * consistency_weight * consistency_loss
            
            # In-place 연산 방지
            loss_consistency_part = self.consistency_coeff * consistency_weight * consistency_loss
            loss = torch.cat([loss[:consistency_batch_size] + loss_consistency_part, loss[consistency_batch_size:]], dim=0)
        
        loss = loss * padding_mask
        return loss, x0_pred, return_sigma, None

#----------------------------------------------------------------------------
