import os
import json
import click
import tqdm
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import save_utils
from training.sampler import inference_edm_sampler as edm_sampler
from training.sampler import padding_mask_from_original_shape
from training.dataset import renewRfProcessedDataset
from denoising_tester import (
    pad_collate_fn,
    dB_to_ratio,
    ratio_to_dB,
    cal_SNR,
    static_noise_generator,
    dynamic_noise_generator,
    boosted_noise_generator,
    parse_int_list,
)


@click.command()
@click.option('--batch', '--batch_size', 'batch_size', default=64, type=click.IntRange(min=1), show_default=True, help='Batch size for testing the sampler.')
@click.option('--network', '--network_pkl', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--config_json', help='Network config json filename', metavar='PATH|URL', type=str, default=None, show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=int, default=11454, show_default=True)
@click.option('--subdirs', help='Create subdirectory for every 1000 seeds', is_flag=True)
@click.option('--output', help='Output directory or file path for saving the CSV results.', type=str, default=None)

@click.option('--data', help='Path to the dataset', metavar='ZIP|DIR', type=str, required=True)
@click.option('--data_keep_ratio', help='How much data keeping ratio', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--must_contain', help='Dataset name should contain', metavar='STR', type=str, default=None, show_default=True)
@click.option('--must_not_contain', help='Dataset name should not contain', metavar='STR', type=str, default=None, show_default=True)
@click.option('--flip_dataset', help='Whether to flip the dataset to use the removed data', is_flag=True)
@click.option('--padding_power2', is_flag=True)

@click.option('--noise_type', help='Noise distribution type for optional condition noise', metavar='static|dynamic|boosted', type=click.Choice(['static', 'dynamic', 'boosted']), default='static', show_default=True)
@click.option('--cond_SNR_dB', 'cond_snr_db', help='Optional SNR (dB) to add noise to Downlink condition (labels). None = clean condition.', type=float, default=None, show_default=True)
@click.option('--additive_noise_sigma', help='Optional sigma for additive noise on Downlink condition (labels).', type=float, default=None, show_default=True)
@click.option('--multiply_noise_sigma', help='Sigma for multiply noise (legacy option)', type=float, default=None, show_default=True)
@click.option('--only_additive_noise', help='Whether to only use additive noise for corruption without natural noise.', is_flag=True)

@click.option('--force_no_nn_scheduler', help='Force noise scheduling to operate in no_nn_scheduler mode (a=0, b=0, d=1) even if model has noise_scheduler', is_flag=True)
@click.option('--scheduler_mode', help='Override scheduler mode for the network', metavar='STR', type=str, default=None, show_default=True)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT', type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min', help='Lowest noise level  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max', help='Highest noise level  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--solver', help='ODE solver', metavar='heun|euler', type=click.Choice(['heun', 'euler']), default='heun', show_default=True)
@click.option('--S_churn', 'S_churn', help='Stochasticity strength', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min', help='Stoch. min noise level', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max', help='Stoch. max noise level', metavar='FLOAT', type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise', help='Stoch. noise inflation', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--stop_sigma', help='Early stop generation at this variance', type=float, default=0.0, show_default=True)
@click.option('--cdf_points', help='Number of quantile points to save in the CDF CSV (0 to save all samples).', type=click.IntRange(min=0), default=1001, show_default=True)
@click.option('--data_norm', help='Data normalization value for the RF data.', type=float, default=1.0, show_default=True)
@click.option('--allow_tf32', is_flag=True)
@click.option('--bf16', help='Enable bfloat16 mixed precision for model computation', is_flag=True)
def main(**kwargs):
    dist.init()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = dnnlib.EasyDict(kwargs)

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if opt.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    noise_type = opt.noise_type
    if noise_type == 'static':
        noise_generator = static_noise_generator
    elif noise_type == 'dynamic':
        noise_generator = dynamic_noise_generator
    elif noise_type == 'boosted':
        noise_generator = boosted_noise_generator
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    data_norm = opt.data_norm

    # Load network.
    dist.print0(f'Loading network from "{opt.network_pkl}"...')

    if "pkl" in opt.network_pkl:
        data = save_utils.load_pkl(opt.network_pkl, verbose=(dist.get_rank() == 0))
        net = data['ema'].to(device)
        if opt.config_json is not None:
            with open(opt.config_json, "r", encoding="utf-8") as f:
                train_opts = json.load(f)
        else:
            config_filepath = os.path.join(os.path.dirname(opt.network_pkl), 'training_options.json')
            assert os.path.isfile(config_filepath), f'Cannot find config file at {config_filepath}'
            dist.print0(f'Loading config from "{config_filepath}"...')
            with open(config_filepath, "r", encoding="utf-8") as f:
                train_opts = json.load(f)

        # Restore / configure scheduler_mode
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
            if hasattr(net.module.noise_scheduler, 'noise_decoder'):
                net.module.noise_scheduler.noise_decoder.scheduler_mode = target_mode

        if opt.force_no_nn_scheduler:
            dist.print0('Forcing noise scheduler to "no_nn_scheduler" mode for noise scheduling.')
            net.scheduler_mode = 'no_nn_scheduler'
            if hasattr(net, 'noise_scheduler'):
                net.noise_scheduler.scheduler_mode = 'no_nn_scheduler'
                if hasattr(net.noise_scheduler, 'noise_decoder'):
                    net.noise_scheduler.noise_decoder.scheduler_mode = 'no_nn_scheduler'
            elif hasattr(getattr(net, 'module', None), 'noise_scheduler'):
                net.module.noise_scheduler.scheduler_mode = 'no_nn_scheduler'
                if hasattr(net.module.noise_scheduler, 'noise_decoder'):
                    net.module.noise_scheduler.noise_decoder.scheduler_mode = 'no_nn_scheduler'
    else:
        print("non pkl file is not supported yet.")
        exit(1)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dataset_kwargs = dnnlib.EasyDict(**train_opts['dataset_kwargs'])
    dataset_kwargs.path = opt.data
    dataset_kwargs.dataset_keep_percentage = opt.data_keep_ratio
    dataset_kwargs.must_contain = opt.must_contain
    dataset_kwargs.must_not_contain = opt.must_not_contain
    if opt.flip_dataset:
        dataset_kwargs.flip_keep_dataset = True

    # Ensure GT Uplink (image) and Downlink (label) are loaded cleanly without dataset-level corruption
    dataset_kwargs.corruption_probability_per_image = 0.0
    dataset_kwargs.additive_noise_sigma = 0.0
    dataset_kwargs.multiply_noise_sigma = 1.0
    dataset_kwargs.only_additive_noise = False
    dataset_kwargs.noise_mean_flag = False
    dataset_kwargs.snr_range = None

    # Deterministically expand all T frames across all files (N_files * T) for full evaluation
    dataset_kwargs.expand_all_frames = True

    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=4, prefetch_factor=2)
    data_loader_kwargs.collate_fn = torch.utils.data.default_collate
    rnd_gen = torch.Generator(device=device).manual_seed(opt.seed)

    dist.print0('Loading dataset...')
    dataset_obj = renewRfProcessedDataset(**dataset_kwargs)
    test_dataset_size = len(dataset_obj)
    dist.print0(f'Total evaluation frames (N_files * T): {test_dataset_size}')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_obj, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )
    dataloader_obj = torch.utils.data.DataLoader(
        dataset=dataset_obj, sampler=dist_sampler, batch_size=opt.batch_size, **data_loader_kwargs
    )

    sampler_kwargs = {
        'num_steps': opt.num_steps,
        'sigma_min': opt.sigma_min,
        'sigma_max': opt.sigma_max,
        'rho': opt.rho,
        'solver': opt.solver,
        'S_churn': opt.S_churn,
        'S_min': opt.S_min,
        'S_max': opt.S_max,
        'S_noise': opt.S_noise,
        'stop_sigma': opt.stop_sigma,
    }
    sampler_kwargs = {k: v for k, v in sampler_kwargs.items() if v is not None}

    num_step_evals = opt.num_steps if opt.solver == 'euler' else max(1, opt.num_steps * 2 - 1)
    predict_SNR_result_sum_dist = torch.tensor(0.0, dtype=torch.float64, device=device)
    predict_SNR_step_sum_dist = torch.zeros(num_step_evals, dtype=torch.float64, device=device)
    local_snr_ratios = []
    local_step_snr_ratios = [[] for _ in range(num_step_evals)]
    local_sample_count = 0

    with torch.inference_mode(True):
        pbar = tqdm.tqdm(
            dataloader_obj,
            total=len(dataloader_obj),
            desc="Uplink Prediction Eval",
            unit="batch",
            dynamic_ncols=True,
            disable=(dist.get_rank() != 0),
        )
        for dataset_item in pbar:
            true_signal = dataset_item["image"].to(device)
            labels = dataset_item["label"].to(device)
            current_sigma = dataset_item["sigma"].to(device)

            normalized_true_signal = true_signal / data_norm
            normalized_labels = labels / data_norm
            normalized_current_sigma = current_sigma / data_norm

            complex_axis = None
            if 'axis_name' in dataset_item:
                for idx, axis_name in enumerate(dataset_item['axis_name']):
                    axis_str = axis_name[0] if isinstance(axis_name, (list, tuple)) else axis_name
                    if isinstance(axis_str, str) and ('complex' in axis_str):
                        complex_axis = idx
                        break
            if complex_axis is None and dataset_kwargs.get('complex_merge_axis') is not None:
                complex_axis = int(dataset_kwargs.complex_merge_axis)

            # Optional noise injection into Downlink Condition (labels) only
            cond_labels = normalized_labels
            if opt.cond_snr_db is not None:
                if complex_axis is not None:
                    batched_complex_axis = complex_axis + 1
                    real_part, imag_part = torch.chunk(normalized_labels, 2, dim=batched_complex_axis)
                    complex_cond = torch.complex(real_part, imag_part)
                else:
                    complex_cond = normalized_labels
                cond_power = torch.mean(
                    torch.real(complex_cond * torch.conj(complex_cond)),
                    dim=tuple(range(1, complex_cond.ndim))
                )
                ratio_snr = dB_to_ratio(opt.cond_snr_db)
                if complex_axis is not None:
                    sigma_cond = (1.0 / (ratio_snr * 2.0) * cond_power) ** 0.5
                else:
                    sigma_cond = (1.0 / ratio_snr * cond_power) ** 0.5
                cond_labels, _, _ = noise_generator(normalized_labels, normalized_current_sigma, sigma_cond)
            elif opt.additive_noise_sigma is not None and opt.additive_noise_sigma > 0.0:
                sigma_cond = torch.full(
                    (normalized_labels.shape[0],),
                    float(opt.additive_noise_sigma) / data_norm,
                    device=device,
                    dtype=normalized_labels.dtype,
                )
                cond_labels, _, _ = noise_generator(normalized_labels, normalized_current_sigma, sigma_cond)

            latents = torch.randn(
                normalized_true_signal.shape,
                generator=rnd_gen,
                device=device,
                dtype=normalized_true_signal.dtype,
            )

            if opt.padding_power2:
                padded_latents, padded_labels, original_shape = pad_collate_fn(latents, cond_labels)
                target_shape = (1,) + tuple(padded_latents.shape[1:])
                input_original_shape = original_shape[0:1]
                padding_mask = padding_mask_from_original_shape(input_original_shape, target_shape)
            else:
                padded_latents = latents
                padded_labels = cond_labels
                original_shape = [latents.shape[1:] for _ in range(latents.shape[0])]
                padding_mask = 1

            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=opt.bf16):
                gen_data, gen_data_each_list = edm_sampler(
                    net,
                    latents=padded_latents,
                    class_labels=padded_labels,
                    padding_mask=padding_mask,
                    **sampler_kwargs
                )

            mask_slice = (slice(0, gen_data.shape[0]),) + tuple(slice(0, int(dim)) for dim in original_shape[0])
            gen_data = gen_data[mask_slice].to(true_signal.dtype) * data_norm

            SNR_ratio = cal_SNR(gen_data, true_signal, complex_axis=complex_axis)
            snr_ratio_f64 = SNR_ratio.detach().to(torch.float64)
            batch_snr_sum = torch.sum(snr_ratio_f64)
            predict_SNR_result_sum_dist += batch_snr_sum
            local_snr_ratios.append(snr_ratio_f64.cpu())
            local_sample_count += int(true_signal.shape[0])

            for i, gen_data_step in enumerate(gen_data_each_list):
                if i >= num_step_evals:
                    break
                gen_data_step = gen_data_step[mask_slice].to(true_signal.dtype) * data_norm
                SNR_step_ratio = cal_SNR(gen_data_step, true_signal, complex_axis=complex_axis)
                snr_step_f64 = SNR_step_ratio.detach().to(torch.float64)
                predict_SNR_step_sum_dist[i] += torch.sum(snr_step_f64)
                local_step_snr_ratios[i].append(snr_step_f64.cpu())

            if dist.get_rank() == 0 and local_sample_count > 0:
                batch_snr_db = float(ratio_to_dB(max((batch_snr_sum / true_signal.shape[0]).item(), 1e-12)))
                running_snr_db = float(ratio_to_dB(max((predict_SNR_result_sum_dist / local_sample_count).item(), 1e-12)))
                pbar.set_postfix(
                    samples=local_sample_count,
                    batch_SNR=f"{batch_snr_db:.2f}dB",
                    avg_SNR=f"{running_snr_db:.2f}dB",
                )

        # Gather per-sample SNR values across all ranks for exact average and CDF computation
        local_snr_tensor = torch.cat(local_snr_ratios, dim=0).to(device) if local_snr_ratios else torch.empty(0, dtype=torch.float64, device=device)
        active_step_lists = [step_list for step_list in local_step_snr_ratios if len(step_list) > 0]
        actual_num_steps = len(active_step_lists)
        if actual_num_steps > 0:
            local_step_tensor = torch.stack([torch.cat(step_list, dim=0) for step_list in active_step_lists], dim=0).to(device)
        else:
            local_step_tensor = torch.empty((0, local_snr_tensor.shape[0]), dtype=torch.float64, device=device)

        if dist.get_world_size() > 1:
            gathered_snr = [torch.zeros_like(local_snr_tensor) for _ in range(dist.get_world_size())]
            torch.distributed.all_gather(gathered_snr, local_snr_tensor)
            # DistributedSampler(shuffle=False) interleaves indices across ranks: [rank, rank + world_size, ...]
            all_snr_ratios_tensor = torch.stack(gathered_snr, dim=1).reshape(-1)[:test_dataset_size]

            if actual_num_steps > 0:
                gathered_steps = [torch.zeros_like(local_step_tensor) for _ in range(dist.get_world_size())]
                torch.distributed.all_gather(gathered_steps, local_step_tensor)
                all_step_snr_ratios_tensor = torch.stack(gathered_steps, dim=2).reshape(actual_num_steps, -1)[:, :test_dataset_size]
            else:
                all_step_snr_ratios_tensor = torch.empty((0, test_dataset_size), dtype=torch.float64, device=device)
        else:
            all_snr_ratios_tensor = local_snr_tensor[:test_dataset_size]
            all_step_snr_ratios_tensor = local_step_tensor[:, :test_dataset_size]

        if dist.get_rank() == 0:
            all_snr_ratios_np = all_snr_ratios_tensor.cpu().numpy()
            all_snr_db_np = ratio_to_dB(np.maximum(all_snr_ratios_np, 1e-12))

            mean_snr_ratio = float(np.mean(all_snr_ratios_np))
            out_snr_db = float(ratio_to_dB(max(mean_snr_ratio, 1e-12)))

            all_step_snr_ratios_np = all_step_snr_ratios_tensor.cpu().numpy()
            all_step_snr_db_np = ratio_to_dB(np.maximum(all_step_snr_ratios_np, 1e-12))
            mean_step_snr_ratio = np.mean(all_step_snr_ratios_np, axis=1) if actual_num_steps > 0 else np.array([])
            out_step_snr_db = ratio_to_dB(np.maximum(mean_step_snr_ratio, 1e-12)) if actual_num_steps > 0 else np.array([])

            # Sort per-sample values for Empirical CDF (ECDF)
            num_total = len(all_snr_db_np)
            sorted_snr_db = np.sort(all_snr_db_np)
            sorted_snr_ratio = np.sort(all_snr_ratios_np)
            sorted_step_snr_db = np.sort(all_step_snr_db_np, axis=1) if actual_num_steps > 0 else np.empty((0, num_total))
            ecdf_prob = np.arange(1, num_total + 1, dtype=np.float64) / float(max(num_total, 1))

            percentiles_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            pct_vals = np.percentile(all_snr_db_np, percentiles_keys)
            pct_dict = {p: float(v) for p, v in zip(percentiles_keys, pct_vals)}
            sample_db_mean = float(np.mean(all_snr_db_np))
            sample_db_std = float(np.std(all_snr_db_np))
            sample_db_min = float(sorted_snr_db[0])
            sample_db_max = float(sorted_snr_db[-1])

            dist.print0("\n=== Uplink Prediction Test Results ===")
            dist.print0(f"Total evaluated samples : {num_total}")
            dist.print0(f"Final SNR average (Ratio): {mean_snr_ratio:>8.4f} ({out_snr_db:>7.4f} dB)")
            dist.print0(
                f"Sample SNR (dB) stats    : Mean={sample_db_mean:.2f} dB, Std={sample_db_std:.2f} dB, "
                f"Median(P50)={pct_dict[50]:.2f} dB, IQR(P25~P75)=[{pct_dict[25]:.2f}, {pct_dict[75]:.2f}] dB, "
                f"P5~P95=[{pct_dict[5]:.2f}, {pct_dict[95]:.2f}] dB"
            )
            dist.print0(f"Final SNR step average (dB): {out_step_snr_db}")

            if opt.output:
                import pandas as pd

                out_path = opt.output
                if out_path.endswith('.csv'):
                    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
                else:
                    os.makedirs(out_path, exist_ok=True)
                    out_path = os.path.join(out_path, 'updownlink_SNR_result.csv')

                res_row = {
                    "Cond_SNR_dB": opt.cond_snr_db if opt.cond_snr_db is not None else "Clean",
                    "Output_SNR_Ratio": mean_snr_ratio,
                    "Output_SNR_dB": out_snr_db,
                    "Sample_SNR_dB_Mean": sample_db_mean,
                    "Sample_SNR_dB_Std": sample_db_std,
                    "Min_SNR_dB": sample_db_min,
                    "P1_SNR_dB": pct_dict[1],
                    "P5_SNR_dB": pct_dict[5],
                    "P10_SNR_dB": pct_dict[10],
                    "P25_SNR_dB": pct_dict[25],
                    "P50_Median_SNR_dB": pct_dict[50],
                    "P75_SNR_dB": pct_dict[75],
                    "P90_SNR_dB": pct_dict[90],
                    "P95_SNR_dB": pct_dict[95],
                    "P99_SNR_dB": pct_dict[99],
                    "Max_SNR_dB": sample_db_max,
                }
                for step_idx, step_db in enumerate(out_step_snr_db):
                    res_row[f"Step_{step_idx}_SNR_dB"] = float(step_db)

                pd.DataFrame([res_row]).to_csv(out_path, index=False)
                dist.print0(f"Saved SNR summary results to {out_path}")

                # Build and save Cumulative Distribution Function (CDF) CSV & NPZ
                if opt.cdf_points == 0 or num_total <= opt.cdf_points:
                    cdf_prob = ecdf_prob
                    cdf_pct = ecdf_prob * 100.0
                    cdf_snr_db = sorted_snr_db
                    cdf_snr_ratio = sorted_snr_ratio
                    cdf_step_snr_db = sorted_step_snr_db
                else:
                    cdf_prob = np.linspace(0.0, 1.0, opt.cdf_points, dtype=np.float64)
                    cdf_pct = cdf_prob * 100.0
                    cdf_snr_db = np.quantile(sorted_snr_db, cdf_prob)
                    cdf_snr_ratio = np.quantile(sorted_snr_ratio, cdf_prob)
                    cdf_step_snr_db = (
                        np.quantile(sorted_step_snr_db, cdf_prob, axis=1).T
                        if actual_num_steps > 0
                        else np.empty((0, opt.cdf_points))
                    )

                cdf_dict = {
                    "Cumulative_Probability": cdf_prob,
                    "Percentile": cdf_pct,
                    "Output_SNR_dB": cdf_snr_db,
                    "Output_SNR_Ratio": cdf_snr_ratio,
                }
                for step_idx in range(actual_num_steps):
                    cdf_dict[f"Step_{step_idx}_SNR_dB"] = cdf_step_snr_db[step_idx]

                cdf_csv_path = out_path.replace('.csv', '_cdf.csv')
                pd.DataFrame(cdf_dict).to_csv(cdf_csv_path, index=False)
                dist.print0(f"Saved CDF curve data ({len(cdf_prob)} points) to {cdf_csv_path}")

                cdf_npz_path = out_path.replace('.csv', '_cdf.npz')
                np.savez_compressed(
                    cdf_npz_path,
                    snr_db=all_snr_db_np,
                    snr_ratio=all_snr_ratios_np,
                    step_snr_db=all_step_snr_db_np,
                    sorted_snr_db=sorted_snr_db,
                    sorted_snr_ratio=sorted_snr_ratio,
                    sorted_step_snr_db=sorted_step_snr_db,
                    ecdf_prob=ecdf_prob,
                    cdf_grid_prob=cdf_prob,
                    cdf_grid_snr_db=cdf_snr_db,
                    cdf_grid_step_snr_db=cdf_step_snr_db,
                )
                dist.print0(f"Saved full per-sample & ECDF arrays (N={num_total}) to {cdf_npz_path}")

                opt_out_path = out_path.replace('.csv', '_opt.json')
                with open(opt_out_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(opt), f, indent=4)
                dist.print0(f"Saved options to {opt_out_path}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
