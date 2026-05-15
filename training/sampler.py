from turtle import st

from matplotlib.pyplot import step
import torch
import numpy as np

def fit_shape(s, t):
    s = torch.tensor(s)
    t = torch.tensor(t)
    while s.ndim < t.ndim:
        s = s.unsqueeze(-1)
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
    sigma_max_exp = sigma_max.unsqueeze(0)
    sigma_min_exp = sigma_min.unsqueeze(0)
    t_steps = (sigma_max_exp ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min_exp ** (1 / rho) - sigma_max_exp ** (1 / rho))) ** rho
    
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

    padding_mask = torch.tensor(padding_mask, device=device)

    # Adjust noise levels based on what's supported by the network.
    if isinstance(sigma_max, torch.Tensor):
        sigma_max = torch.clamp(sigma_max, max=net.sigma_max).to(device)
    else:
        sigma_max = min(sigma_max, net.sigma_max)
        
    if isinstance(sigma_min, torch.Tensor):
        sigma_min = torch.clamp(sigma_min, min=net.sigma_min).to(device)
    else:
        sigma_min = max(sigma_min, net.sigma_min)

    x_list = []

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if isinstance(sigma_max, torch.Tensor):
        for i in range(abs(sigma_max.ndim - step_indices.ndim)):
            step_indices = step_indices.unsqueeze(-1)

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
            
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        step_noise_scale = (t_hat ** 2 - t_cur ** 2).clamp(min=0).sqrt()
        x_hat = (x_cur + fit_shape(step_noise_scale, x_cur) * S_noise * randn_like(x_cur)) * padding_mask

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64) * padding_mask
        x_list.append(denoised.clone().detach())
        
        # Stop if variance is below threshold
        if isinstance(t_next, torch.Tensor) and t_next.ndim > 0:
            if (t_next < stop_sigma).all():
                return denoised, x_list
        else:
            if t_next < stop_sigma:
                return denoised, x_list

        d_cur = (x_hat - denoised) / fit_shape(t_hat, x_cur)
        x_next = x_hat + fit_shape(t_next - t_hat, x_cur) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64) * padding_mask
            x_list.append(denoised.clone().detach())
            d_prime = (x_next - denoised) / fit_shape(t_next, x_cur)
            x_next = x_hat + fit_shape(t_next - t_hat, x_cur) * (0.5 * d_cur + 0.5 * d_prime)
        

    return x_next, x_list
