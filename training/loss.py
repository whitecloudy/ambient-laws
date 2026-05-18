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
from training.sampler import edm_sampler, padding_mask_from_original_shape


def from_x0_pred_to_xnature_pred_ve_to_ve_modify(x0_pred, noisy_input, current_sigma, desired_sigma):
    return (1 - (desired_sigma / current_sigma) ** 2) * x0_pred + ((desired_sigma / current_sigma) ** 2) * noisy_input

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
        sigma = torch.clamp(sigma, min=current_sigma + 1e-5)
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
            new_sigma = (new_rnd_normal * self.P_std + self.P_mean).exp()
            new_sigma = torch.clamp(new_sigma, max=sigma)

            # we will only keep the first batch_size / self.num_primes part of the batch
            consistency_batch_size = self.consistency_batch_size
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
            loss[:consistency_batch_size] += self.consistency_coeff * consistency_weight * consistency_loss
        
        loss = loss * padding_mask
        return loss, x0_pred, return_sigma

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
            loss[:consistency_batch_size] += self.consistency_coeff * consistency_weight * consistency_loss
        
        loss = loss * padding_mask
        return loss, x0_pred, return_sigma

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
            current_sigma = torch.tensor(0.0)
        
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
            loss[:consistency_batch_size] += self.consistency_coeff * consistency_weight * consistency_loss
        
        loss = loss * padding_mask
        return loss, x0_pred, return_sigma

#----------------------------------------------------------------------------
