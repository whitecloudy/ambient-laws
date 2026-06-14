from turtle import st

from matplotlib.pyplot import step
import torch
import numpy as np

def fit_shape(s, t):
    if isinstance(s, torch.Tensor) and isinstance(t, torch.Tensor):
        while s.ndim < t.ndim:
            s = s.unsqueeze(-1)
        return s
    else:
        return s

def padding_mask_from_original_shape(original_shape, target_shape):
    padding_mask = torch.zeros(target_shape, device=original_shape.device)
    for i in range(target_shape[0]):
        slices = (i,) + tuple(slice(0, int(dim)) for dim in original_shape[i])
        padding_mask[slices] = 1
    return padding_mask


def edm_sampler(
    net, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, padding_mask=1):
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device).view(-1, *[1]*latents.ndim)
    # Adjust noise levels based on what's supported by the network.
    if isinstance(sigma_max, torch.Tensor):
        sigma_max = sigma_max.unsqueeze(0)
        sigma_max = sigma_max.expand([num_steps, ]+([-1, ]*(len(sigma_max.shape)-1)))
    else:
        sigma_max = min(sigma_max, net.sigma_max)
        
    if isinstance(sigma_min, torch.Tensor):
        sigma_min = sigma_min.unsqueeze(0)
        sigma_min = sigma_min.expand([num_steps, ]+([-1, ]*(len(sigma_min.shape)-1)))
    else:
        sigma_min = max(sigma_min, net.sigma_min)
        
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    step_indices = fit_shape(step_indices, sigma_max)
    step_indices = fit_shape(step_indices, sigma_min)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    x_next = latents
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + 2 * (t_next - t_cur) * d_cur  + (torch.sqrt(2 * (t_cur - t_next).abs() * t_cur) * torch.randn_like(x_cur) * padding_mask)

    return x_next


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

# Return
# - x_0 : x in step 0
# - x_t list : all x_t list
def inference_edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    stop_sigma=0.0, latents_already_noisy=False, padding_mask=1,
):
    batch_size = latents.shape[0]
    device = latents.device

    if isinstance(padding_mask, float) and padding_mask == 1:
        padding_exist = False
    else:
        padding_exist = True

    padding_mask = torch.tensor(padding_mask, device=device)

    # Adjust noise levels based on what's supported by the network.
    if isinstance(sigma_max, torch.Tensor):
        sigma_max = torch.clamp(sigma_max, max=net.sigma_max).unsqueeze(0).to(device)
        sigma_max = sigma_max.expand([num_steps, ]+([-1, ]*(len(sigma_max.shape)-1)))
    else:
        sigma_max = min(sigma_max, net.sigma_max)
        
    if isinstance(sigma_min, torch.Tensor):
        sigma_min = torch.clamp(sigma_min, min=net.sigma_min).unsqueeze(0).to(device)
        sigma_min = sigma_min.expand([num_steps, ]+([-1, ]*(len(sigma_min.shape)-1)))
    else:
        sigma_min = max(sigma_min, net.sigma_min)

    x_list = []

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    step_indices = fit_shape(step_indices, sigma_max)
    step_indices = fit_shape(step_indices, sigma_min)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])], dim=0) # t_N = 0
    
    # Main sampling loop.
    t_next_0 = t_steps[0]
    
    if latents_already_noisy:
        x_next = latents.to(torch.float64) * padding_mask
    else:
        x_next = latents.to(torch.float64) * fit_shape(t_next_0, latents) * padding_mask
    for i in range(num_steps): # 0, ..., N-1
        x_cur = x_next
        t_cur = t_steps[i]
        t_next = t_steps[i+1]

        # Increase noise temporarily.
        if isinstance(t_cur, torch.Tensor) and t_cur.ndim > 0:
            gamma_val = min(S_churn / num_steps, np.sqrt(2) - 1)
            gamma = torch.where(
                (t_cur >= S_min) & (t_cur <= S_max),
                torch.tensor(gamma_val, device=t_cur.device, dtype=t_cur.dtype),
                torch.tensor(0.0, device=t_cur.device, dtype=t_cur.dtype)
            )
        else:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0

        t_cur_dynamic = t_cur
        t_hat_dynamic = net.round_sigma(t_cur_dynamic + gamma * t_cur_dynamic)
        t_next_dynamic = t_next
        step_noise_scale = (t_hat_dynamic ** 2 - t_cur_dynamic ** 2).clamp(min=0).sqrt()
        step_noise_scale = fit_shape(step_noise_scale, x_cur)

        if isinstance(step_noise_scale, torch.Tensor) and step_noise_scale.shape != x_cur.shape and padding_exist:
            pad_width = []
            for dim_idx in range(x_cur.ndim - 1, -1, -1):
                if step_noise_scale.shape[dim_idx] == 1 or step_noise_scale.shape[dim_idx] == x_cur.shape[dim_idx]:
                    pad_width.extend([0, 0])
                else:
                    pad_width.extend([0, max(0, x_cur.shape[dim_idx] - step_noise_scale.shape[dim_idx])])
            step_noise_scale = torch.nn.functional.pad(step_noise_scale, tuple(pad_width))
        x_hat = (x_cur + step_noise_scale * S_noise * randn_like(x_cur)) * padding_mask

        # Euler step.
        t_hat_net = t_hat_dynamic
        if padding_exist:
            t_hat_net = torch.mean(t_hat_dynamic, dim=list(range(1, t_hat_dynamic.ndim)), keepdim=True)
        elif isinstance(t_hat_net, torch.Tensor) and t_hat_net.ndim == x_hat.ndim:
            if t_hat_net.shape != x_hat.shape:
                t_hat_net = t_hat_net.expand_as(x_hat)

        if isinstance(t_hat_net, torch.Tensor):
            if t_hat_net.ndim == 0:
                t_hat_net = t_hat_net.unsqueeze(0).expand(batch_size)
            elif t_hat_net.shape[0] != batch_size:
                t_hat_net = t_hat_net.expand([batch_size] + list(t_hat_net.shape[1:]))

        denoised = net(x_hat, t_hat_net, class_labels).to(torch.float64) * padding_mask
        x_list.append(denoised.clone().detach())
        
        # Stop if variance is below threshold
        if isinstance(t_next_dynamic, torch.Tensor) and t_next_dynamic.ndim > 0:
            if (t_next_dynamic < stop_sigma).all():
                return denoised, x_list
        else:
            if t_next_dynamic < stop_sigma:
                return denoised, x_list

        d_cur = (x_hat - denoised) / fit_shape(t_hat_net, x_cur)
        step_noise_scale = fit_shape(t_next_dynamic - t_hat_dynamic, x_cur)
        if isinstance(step_noise_scale, torch.Tensor) and step_noise_scale.shape != x_cur.shape and padding_exist:
            pad_width = []
            for dim_idx in range(x_cur.ndim - 1, -1, -1):
                if step_noise_scale.shape[dim_idx] == 1 or step_noise_scale.shape[dim_idx] == x_cur.shape[dim_idx]:
                    pad_width.extend([0, 0])
                else:
                    pad_width.extend([0, max(0, x_cur.shape[dim_idx] - step_noise_scale.shape[dim_idx])])
            step_noise_scale = torch.nn.functional.pad(step_noise_scale, tuple(pad_width))

        x_pred = x_hat + step_noise_scale * d_cur * padding_mask

        # Apply 2nd order correction.
        if i < num_steps - 1:
            t_next_net = t_next_dynamic
            if padding_exist:
                t_next_net = torch.mean(t_next_dynamic, dim=list(range(1, t_next_dynamic.ndim)), keepdim=True)
            elif isinstance(t_next_net, torch.Tensor) and t_next_net.ndim == x_pred.ndim:
                if t_next_net.shape != x_pred.shape:
                    t_next_net = t_next_net.expand_as(x_pred)

            if isinstance(t_next_net, torch.Tensor):
                if t_next_net.ndim == 0:
                    t_next_net = t_next_net.unsqueeze(0).expand(batch_size)
                elif t_next_net.shape[0] != batch_size:
                    t_next_net = t_next_net.expand([batch_size] + list(t_next_net.shape[1:]))
            denoised_prime = net(x_pred, t_next_net, class_labels).to(torch.float64) * padding_mask
            x_list.append(denoised_prime.clone().detach())
            d_prime = (x_pred - denoised_prime) / fit_shape(t_next_net, x_cur)
            
            step_noise_scale = fit_shape(t_next_dynamic - t_hat_dynamic, x_hat)
            if isinstance(step_noise_scale, torch.Tensor) and step_noise_scale.shape != x_hat.shape and padding_exist:
                pad_width = []
                for dim_idx in range(x_cur.ndim - 1, -1, -1):
                    if step_noise_scale.shape[dim_idx] == 1 or step_noise_scale.shape[dim_idx] == x_hat.shape[dim_idx]:
                        pad_width.extend([0, 0])
                    else:
                        pad_width.extend([0, max(0, x_hat.shape[dim_idx] - step_noise_scale.shape[dim_idx])])
                step_noise_scale = torch.nn.functional.pad(step_noise_scale, tuple(pad_width))

            x_next = x_hat + step_noise_scale * (0.5 * d_cur + 0.5 * d_prime)
        else:
            x_next = x_pred

    return x_next, x_list
