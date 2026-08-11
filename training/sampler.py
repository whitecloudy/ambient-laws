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
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device).reshape(-1, *[1]*latents.ndim)
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


def _get_net_attr(n, name):
    if hasattr(n, name):
        return getattr(n, name)
    elif hasattr(n, 'module') and hasattr(n.module, name):
        return getattr(n.module, name)
    return None

def edm_sampler_with_scheduler(
    net, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, padding_mask=1,
    noise_schedule_dict=None, new_sigma_ref=None
):
    get_noise_scheduling_fn = _get_net_attr(net, "get_noise_scheduling")
    if noise_schedule_dict is not None and get_noise_scheduling_fn is not None:
        sigma_ref = noise_schedule_dict.get('sigma_ref', None)
        current_sigma = noise_schedule_dict.get('current_sigma', None)
        z = noise_schedule_dict.get('latent_z', None)
        abd = noise_schedule_dict.get('abd', None)

        if sigma_ref is not None and new_sigma_ref is not None and current_sigma is not None:
            step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
            ref_sigma_max = sigma_ref.to(torch.float64)
            ref_sigma_min = new_sigma_ref.to(torch.float64)
            
            if isinstance(ref_sigma_max, torch.Tensor):
                ref_sigma_max = ref_sigma_max.unsqueeze(0).expand([num_steps] + [-1] * ref_sigma_max.ndim)
            if isinstance(ref_sigma_min, torch.Tensor):
                ref_sigma_min = ref_sigma_min.unsqueeze(0).expand([num_steps] + [-1] * ref_sigma_min.ndim)
            
            step_indices = fit_shape(step_indices, ref_sigma_max)
            step_indices = fit_shape(step_indices, ref_sigma_min)
            
            ref_t_steps = (ref_sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (ref_sigma_min ** (1 / rho) - ref_sigma_max ** (1 / rho))) ** rho
            
            t_steps = []
            for i in range(num_steps):
                ref_t = ref_t_steps[i]
                ref_t_in = ref_t.to(dtype=latents.dtype) if isinstance(ref_t, torch.Tensor) else ref_t
                sched_res = get_noise_scheduling_fn(ref_t_in, current_sigma, z=z, abd=abd)
                t_steps.append(sched_res['poly_sigma'])
        else:
            step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
            if isinstance(sigma_max, torch.Tensor):
                sigma_max_t = sigma_max.unsqueeze(0).expand([num_steps, ]+([-1, ]*(len(sigma_max.shape)-1)))
            else:
                sigma_max_t = min(sigma_max, getattr(net, 'sigma_max', 80))
                
            if isinstance(sigma_min, torch.Tensor):
                sigma_min_t = sigma_min.unsqueeze(0).expand([num_steps] + [-1] * len(sigma_min.shape))
            else:
                sigma_min_t = max(sigma_min, getattr(net, 'sigma_min', 0.002))
                
            step_indices = fit_shape(step_indices, sigma_max_t)
            step_indices = fit_shape(step_indices, sigma_min_t)
            t_steps_tensor = (sigma_max_t ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min_t ** (1 / rho) - sigma_max_t ** (1 / rho))) ** rho
            t_steps = [t_steps_tensor[i] for i in range(num_steps)]
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        if isinstance(sigma_max, torch.Tensor):
            sigma_max_t = sigma_max.unsqueeze(0).expand([num_steps] + [-1] * len(sigma_max.shape))
        else:
            sigma_max_t = min(sigma_max, getattr(net, 'sigma_max', 80))
            
        if isinstance(sigma_min, torch.Tensor):
            sigma_min_t = sigma_min.unsqueeze(0).expand([num_steps] + [-1] * len(sigma_min.shape))
        else:
            sigma_min_t = max(sigma_min, getattr(net, 'sigma_min', 0.002))
            
        step_indices = fit_shape(step_indices, sigma_max_t)
        step_indices = fit_shape(step_indices, sigma_min_t)
        t_steps_tensor = (sigma_max_t ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min_t ** (1 / rho) - sigma_max_t ** (1 / rho))) ** rho
        t_steps = [t_steps_tensor[i] for i in range(num_steps)]

    x_next = latents
    for i in range(num_steps - 1): # 0, ..., N-2
        x_cur = x_next
        t_cur = t_steps[i]
        t_next = t_steps[i+1]

        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        t_cur_fitted = fit_shape(t_cur, x_cur)
        d_cur = (x_cur - denoised) / t_cur_fitted
        step_diff = t_next - t_cur
        x_next = x_cur + 2 * step_diff * d_cur + (torch.sqrt(2 * step_diff.abs() * t_cur_fitted) * torch.randn_like(x_cur) * padding_mask)

    return x_next


#----------------------------------------------------------------------------
# Proposed EDM sampler with Noise Scheduler.

def inference_edm_sampler_with_scheduler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    stop_sigma=0.0, latents_already_noisy=False, padding_mask=1,
    sigma_t_n=None, z=None,
):
    batch_size = latents.shape[0]
    device = latents.device

    if isinstance(padding_mask, float) and padding_mask == 1:
        padding_exist = False
    else:
        padding_exist = True

    padding_mask = torch.as_tensor(padding_mask, device=device)

    get_noise_scheduling_fn = _get_net_attr(net, "get_noise_scheduling")
    generate_latent_z_fn = _get_net_attr(net, "generate_latent_z")

    # Adjust noise levels based on what's supported by the network.
    net_sigma_max = getattr(net, 'sigma_max', 80)
    net_sigma_min = getattr(net, 'sigma_min', 0.002)

    raw_sigma_max = torch.clamp(sigma_max, max=net_sigma_max) if isinstance(sigma_max, torch.Tensor) else min(sigma_max, net_sigma_max)
    raw_sigma_min = torch.clamp(sigma_min, min=net_sigma_min) if isinstance(sigma_min, torch.Tensor) else max(sigma_min, net_sigma_min)

    if isinstance(raw_sigma_max, torch.Tensor):
        sigma_max = raw_sigma_max.unsqueeze(0).to(device)
        sigma_max = sigma_max.expand([num_steps] + ([-1] * (len(sigma_max.shape) - 1)))
    else:
        sigma_max = raw_sigma_max
        
    if isinstance(raw_sigma_min, torch.Tensor):
        sigma_min = raw_sigma_min.unsqueeze(0).to(device)
        sigma_min = sigma_min.expand([num_steps] + ([-1] * (len(sigma_min.shape) - 1)))
    else:
        sigma_min = raw_sigma_min

    # Determine sigma_t_n input:
    # latents_already_noisy == False -> sigma_min
    # latents_already_noisy == True -> initial latents' sigma_t_n (passed sigma_t_n or sigma_max)
    if not latents_already_noisy:
        sigma_t_n_input = raw_sigma_min
    else:
        if sigma_t_n is not None:
            sigma_t_n_input = sigma_t_n
        else:
            sigma_t_n_input = raw_sigma_max

    if not isinstance(sigma_t_n_input, torch.Tensor):
        sigma_t_n_tensor = torch.full((batch_size,), float(sigma_t_n_input), device=device, dtype=latents.dtype)
    else:
        sigma_t_n_tensor = sigma_t_n_input.to(device=device, dtype=latents.dtype)
        if sigma_t_n_tensor.ndim == 0:
            sigma_t_n_tensor = sigma_t_n_tensor.unsqueeze(0).expand(batch_size)
    while sigma_t_n_tensor.ndim < latents.ndim:
        sigma_t_n_tensor = sigma_t_n_tensor.unsqueeze(-1)
    sigma_t_n_tensor = sigma_t_n_tensor.expand_as(latents)

    # Determine z input:
    # latents_already_noisy == False -> generated randomly inside net (z=None)
    # latents_already_noisy == True -> generated from initial latent input via net encoder
    if z is None:
        if latents_already_noisy:
            if generate_latent_z_fn is not None:
                z_input, _ = generate_latent_z_fn(latents, sigma_t_n_tensor)
            else:
                z_input = None
        else:
            z_input = None
    else:
        z_input = z

    x_list = []

    # Time step discretization in reference space.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    step_indices = fit_shape(step_indices, sigma_max)
    step_indices = fit_shape(step_indices, sigma_min)
    t_ref_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    round_sigma_fn = _get_net_attr(net, "round_sigma")
    if round_sigma_fn is not None:
        t_ref_steps = torch.cat([round_sigma_fn(t_ref_steps), torch.zeros_like(t_ref_steps[:1])], dim=0) # t_N = 0
    else:
        t_ref_steps = torch.cat([t_ref_steps, torch.zeros_like(t_ref_steps[:1])], dim=0)

    abd = None

    def get_sched_sigma(ref_t):
        nonlocal abd, z_input
        if ref_t is None or (isinstance(ref_t, (int, float)) and ref_t == 0) or (isinstance(ref_t, torch.Tensor) and (ref_t == 0).all()):
            return torch.zeros_like(latents, dtype=torch.float64)
        
        if isinstance(ref_t, torch.Tensor):
            ref_t_in = ref_t.to(dtype=latents.dtype)
            if ref_t_in.ndim == 0:
                ref_t_in = ref_t_in.unsqueeze(0).expand(batch_size)
        else:
            ref_t_in = torch.full((batch_size,), float(ref_t), device=device, dtype=latents.dtype)

        if get_noise_scheduling_fn is not None:
            res = get_noise_scheduling_fn(ref_t_in, sigma_t_n_tensor, z=z_input, abd=abd)
            abd = res.get('abd', abd)
            if z_input is None and 'latent_z' in res:
                z_input = res['latent_z']
            return res['poly_sigma'].to(torch.float64)
        else:
            return torch.as_tensor(ref_t, dtype=torch.float64, device=device)

    # Initial latent setup.
    if latents_already_noisy:
        x_next = latents.to(torch.float64) * padding_mask
    else:
        sigma_0 = get_sched_sigma(t_ref_steps[0])
        x_next = latents.to(torch.float64) * fit_shape(sigma_0, latents) * padding_mask

    for i in range(num_steps): # 0, ..., N-1
        x_cur = x_next
        t_cur_ref = t_ref_steps[i]
        t_next_ref = t_ref_steps[i+1]

        sigma_cur = get_sched_sigma(t_cur_ref)

        if i == num_steps - 1 or (isinstance(t_next_ref, torch.Tensor) and (t_next_ref == 0).all()) or (not isinstance(t_next_ref, torch.Tensor) and t_next_ref == 0):
            sigma_next = torch.zeros_like(x_cur)
        else:
            sigma_next = get_sched_sigma(t_next_ref)

        # Increase noise temporarily (Churn).
        if isinstance(t_cur_ref, torch.Tensor) and t_cur_ref.ndim > 0:
            gamma_val = min(S_churn / num_steps, np.sqrt(2) - 1)
            gamma = torch.where(
                (t_cur_ref >= S_min) & (t_cur_ref <= S_max),
                torch.tensor(gamma_val, device=t_cur_ref.device, dtype=t_cur_ref.dtype),
                torch.tensor(0.0, device=t_cur_ref.device, dtype=t_cur_ref.dtype)
            )
        else:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur_ref <= S_max else 0

        t_hat_ref = t_cur_ref + gamma * t_cur_ref
        if round_sigma_fn is not None:
            t_hat_ref = round_sigma_fn(t_hat_ref)

        sigma_hat = get_sched_sigma(t_hat_ref)

        step_noise_scale = (sigma_hat ** 2 - sigma_cur ** 2).clamp(min=0).sqrt()
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
        class_labels_to_pass = z_input if class_labels is None else class_labels
        denoised = net(x_hat, sigma_hat, class_labels=class_labels_to_pass, z=z_input).to(torch.float64) * padding_mask
        x_list.append(denoised.clone().detach())

        # Stop if variance is below threshold.
        if isinstance(t_next_ref, torch.Tensor) and t_next_ref.ndim > 0:
            if (t_next_ref < stop_sigma).all():
                return denoised, x_list
        else:
            if t_next_ref < stop_sigma:
                return denoised, x_list

        safe_sigma_hat = torch.where(sigma_hat == 0, torch.tensor(1e-8, device=sigma_hat.device, dtype=sigma_hat.dtype), sigma_hat)
        d_cur = (x_hat - denoised) / fit_shape(safe_sigma_hat, x_cur)
        step_noise_scale = fit_shape(sigma_next - sigma_hat, x_cur)
        if isinstance(step_noise_scale, torch.Tensor) and step_noise_scale.shape != x_cur.shape and padding_exist:
            pad_width = []
            for dim_idx in range(x_cur.ndim - 1, -1, -1):
                if step_noise_scale.shape[dim_idx] == 1 or step_noise_scale.shape[dim_idx] == x_cur.shape[dim_idx]:
                    pad_width.extend([0, 0])
                else:
                    pad_width.extend([0, max(0, x_cur.shape[dim_idx] - step_noise_scale.shape[dim_idx])])
            step_noise_scale = torch.nn.functional.pad(step_noise_scale, tuple(pad_width))

        x_pred = x_hat + step_noise_scale * d_cur * padding_mask

        # Apply 2nd order correction (Heun).
        if i < num_steps - 1:
            denoised_prime = net(x_pred, sigma_next, class_labels=class_labels_to_pass, z=z_input).to(torch.float64) * padding_mask
            x_list.append(denoised_prime.clone().detach())
            safe_sigma_next = torch.where(sigma_next == 0, torch.tensor(1e-8, device=sigma_next.device, dtype=sigma_next.dtype), sigma_next)
            d_prime = (x_pred - denoised_prime) / fit_shape(safe_sigma_next, x_cur)

            step_noise_scale = fit_shape(sigma_next - sigma_hat, x_hat)
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
    **kwargs
):
    if _get_net_attr(net, "get_noise_scheduling") is not None:
        return inference_edm_sampler_with_scheduler(
            net=net, latents=latents, class_labels=class_labels, randn_like=randn_like,
            num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise,
            stop_sigma=stop_sigma, latents_already_noisy=latents_already_noisy, padding_mask=padding_mask,
            **kwargs
        )

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

