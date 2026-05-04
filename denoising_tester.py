import numpy as np
import torch
import click
import dnnlib
from torch_utils import distributed as dist
import pickle
import os
import json
import tqdm
from training.sampler import inference_edm_sampler as edm_sampler
from training.sampler import padding_mask_from_original_shape
from training.dataset import  renewRfProcessedDataset

def _power2ceil(original):
    return int(2**np.ceil(np.log2(original)))

def pad_collate_fn(img, label=None):
    """
    default_collate_fn을 통과하여 텐서 배치 형태(dict)로 묶인 데이터를
    2의 거듭제곱 크기로 패딩합니다.
    """

    # 배치로 묶인 텐서의 마지막 두 차원을 H, W로 간주 (N, C, H, W)
    h, w = img.shape[-2], img.shape[-1]

    target_h = _power2ceil(h)
    target_w = _power2ceil(w)

    pad_h = target_h - h
    pad_w = target_w - w

    # 패딩 전 원래의 형태 정보(C, H, W)를 배치 사이즈(N)만큼 생성하여 저장
    original_shape = torch.tensor(img.shape[1:], dtype=torch.long, device=img.device)
    original_shape = original_shape.unsqueeze(0).expand(img.shape[0], -1)

    if pad_h > 0 or pad_w > 0:
        # torch.nn.functional.pad는 뒤에서부터 (left, right, top, bottom) 순서로 적용
        pad_width = (0, pad_w, 0, pad_h)
        img = torch.nn.functional.pad(img, pad_width, mode='constant', value=0)

        # image와 label의 shape이 같으면(예: segmentation) label도 동일하게 패딩
        if label is not None and img.shape == label.shape:
            label = torch.nn.functional.pad(label, pad_width, mode='constant', value=0)

    return img, label, original_shape


def dB_to_ratio(dB):
    return 10 ** (dB / 10)

def ratio_to_dB(ratio):
    return 10 * np.log10(ratio)


def _match_axis(source: torch.Tensor, target: torch.Tensor):
    if source.ndim >= target.ndim:
        # print(f"Warning: we cannot match axis if source ndim (={source.ndim}) is larger than target ndim (={target.ndim}). ")
        return source
    else:
        return_tensor = source
        for i in range(target.ndim - source.ndim):
            return_tensor = return_tensor.unsqueeze(-1)
        return return_tensor
    
def cal_SNR(predict : torch.Tensor, truth : torch.Tensor, complex_axis=None):
    if complex_axis != None:
        assert predict.dtype != torch.complex and truth.dtype != torch.complex, "\'complex axis\' is given while dtype is already complex!"
        # Handling batch axis
        if complex_axis >= 0:
            complex_axis += 1

        predict_split = torch.split(predict, 2, dim=complex_axis)
        truth_split = torch.split(truth, 2, dim=complex_axis)

        predict = predict_split[0] + 1j * predict_split[1]
        truth = truth_split[0] + 1j * truth_split[1]
    axis_list = [i for i in range(1, len(predict.shape))]
    PS = torch.sum(torch.abs(truth)**2, dim=axis_list)  # power of signal
    PN = torch.sum(torch.abs(predict - truth)**2, dim=axis_list)  # power of noise
    ratio = PS / PN
    return ratio


#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
import re

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



@click.command()
@click.option('--batch_size', default=64, help='Batch size for testing the sampler.')
@click.option('--network_pkl', help='Network pickle filename', metavar='PATH|URL',                                  type=str, required=True)
@click.option('--config_json',              help='Network config json filename', metavar='PATH|URL',                type=str, default=None, show_default=True)
@click.option('--seed',                     help='Random seed', metavar='INT',                                      type=int, default=11454, show_default=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--output',                   help='Output directory or file path for saving the CSV results.',       type=str, default=None)

@click.option('--data',                     help='Path to the dataset', metavar='ZIP|DIR',                          type=str, required=True)
@click.option('--data_keep_ratio',          help='How much data keeping ratio', metavar='FLOAT',                    type=float, default=1.0, show_default=True)
@click.option('--must_contain',             help='Dataset name should contain', metavar='STR',                      type=str, default=None, show_default=True)
@click.option('--must_not_contain',         help='Dataset name should not contain', metavar='STR',                  type=str, default=None, show_default=True)
@click.option('--flip_dataset',             help='Whether to flip the dataset to use the removed data',             is_flag=True)
@click.option('--test_SNR_range', 'test_SNR_range', help='SNR range will be tested', metavar='STR',                 type=str, default='5,20', show_default=True)
@click.option('--test_SNR_step', 'test_SNR_step',   help='SNR step will be tested',                                 type=float, default=1.0, show_default=True)
@click.option('--same_sigma_level',         help='Whether use same sigma level across all signal matrix',           is_flag=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--data_norm',               help='Data normalization value for the RF data.',                        type=float, default=1.0, show_default=True)
def main(**kwargs):
    dist.init()

    device=torch.device('cuda')
    opt = dnnlib.EasyDict(kwargs)

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    data_norm = opt.data_norm


    # Load network.
    dist.print0(f'Loading network from "{opt.network_pkl}"...')

    if "pkl" in opt.network_pkl:
        with dnnlib.util.open_url(opt.network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
        if opt.config_json is not None:
            with open(opt.config_json, "r", encoding="utf-8") as f:
                train_opts = json.load(f)
        else:
            config_filepath = os.path.join(os.path.dirname(opt.network_pkl), 'training_options.json')
            assert os.path.isfile(config_filepath), f'Cannot find config file at {config_filepath}'
            dist.print0(f'Loading config from "{config_filepath}"...')
            with open(config_filepath, "r", encoding="utf-8") as f:
                train_opts = json.load(f)
    else:
        print("non pkl file is not supported yet.")
        exit(1)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # opts = opts['dataset_kwargs']
    dataset_kwargs = dnnlib.EasyDict(**train_opts['dataset_kwargs'])
    dataset_kwargs.path = opt.data
    dataset_kwargs.corruption_probability_per_image = 0.0   # We will add noise in this code
    dataset_kwargs.dataset_keep_percentage = opt.data_keep_ratio
    dataset_kwargs.must_contain = opt.must_contain
    dataset_kwargs.must_not_contain = opt.must_not_contain
    if opt.flip_dataset:
        dataset_kwargs.flip_keep_dataset = True

    # We will add noise in this code
    dataset_kwargs.additive_noise_sigma = 0.0
    dataset_kwargs.multiply_noise_sigma = 1.0
    dataset_kwargs.only_additive_noise = False

    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=8, prefetch_factor=4)
        
    data_loader_kwargs.collate_fn = torch.utils.data.default_collate
    rnd_gen = torch.Generator(device=device).manual_seed(opt.seed)

    dist.print0('Loading dataset...')

    dataset_obj = renewRfProcessedDataset(**dataset_kwargs)
    test_dataset_size = len(dataset_obj)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset_obj, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False) # type: ignore
    dataloader_obj = torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dist_sampler, batch_size=opt.batch_size, **data_loader_kwargs)
    test_SNR_range = opt.test_SNR_range
    test_SNR_range = [float(x) for x in test_SNR_range.split(',')]
    test_SNR_step = opt.test_SNR_step

    SNR_steps = np.arange(test_SNR_range[0], test_SNR_range[1] + test_SNR_step, test_SNR_step)
    ratio_SNR_steps = dB_to_ratio(SNR_steps)

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

    predict_SNR_result_dict = {}

    with torch.inference_mode(True):
        predict_SNR_result_sum_dist_list = torch.zeros(ratio_SNR_steps.shape[0], device=device)
        for dataset_item in tqdm.tqdm(dataloader_obj, unit='iter', disable=(dist.get_rank() != 0)):
            true_signal = dataset_item["image"].to(device)
            # labels = dataset_item["label"].to(device)
            current_sigma = dataset_item["sigma"].to(device)

            complex_axis = None
            complex_axis_name = None
            if 'axis_name' in dataset_item:
                for idx, axis_name in enumerate(dataset_item['axis_name']):
                    # default_collate에 의해 문자열 리스트가 튜플로 묶일 수 있으므로 첫 번째 요소 추출
                    axis_str = axis_name[0] if isinstance(axis_name, (list, tuple)) else axis_name
                    if isinstance(axis_str, str) and ('complex' in axis_str):
                        complex_axis = idx
                        complex_axis_name = axis_str
                        break
            
            if complex_axis is not None:
                # DataLoader를 거치며 맨 앞에 배치 차원(batch dimension)이 추가되었으므로 +1
                batched_complex_axis = complex_axis + 1
                
                # 해당 축을 기준으로 실수부(real)와 허수부(imag)로 절반씩 나눕니다.
                real_part, imag_part = torch.chunk(true_signal, 2, dim=batched_complex_axis)
                
                # 실수부와 허수부를 합쳐서 복소수 텐서로 재구성합니다.
                complex_signal = torch.complex(real_part, imag_part)
            else:
                # complex 축이 지정되지 않은 경우 기존 신호를 그대로 할당합니다.
                complex_signal = true_signal

            complex_signal = complex_signal / data_norm

            # (Batch)
            signal_power = torch.mean(torch.real(complex_signal * torch.conj(complex_signal)), dim=tuple(range(1, complex_signal.ndim)))

            normalized_current_sigma = current_sigma / torch.mean(current_sigma**2, dim=tuple(range(1, current_sigma.ndim)))
            
            for idx, ratio_SNR in enumerate(ratio_SNR_steps):
                # (Batch)
                sigma_SNR_steps = (1/ratio_SNR * signal_power) ** 0.5
                if not opt.same_sigma_level:
                    sigma_SNR_steps = _match_axis(sigma_SNR_steps, normalized_current_sigma)
                    input_sigma = normalized_current_sigma * sigma_SNR_steps
                else:
                    input_sigma = sigma_SNR_steps

                input_signal = true_signal / data_norm

                multiply_sigma = _match_axis(input_sigma, input_signal)

                input_signal = input_signal + torch.randn_like(input_signal) * multiply_sigma

                padded_signal, _, original_shape = pad_collate_fn(input_signal)

                target_shape = (1,) + tuple(padded_signal.shape[1:])
                input_original_shape = torch.tensor(tuple(original_shape[0]))
                input_original_shape = input_original_shape.unsqueeze(0) # add batch dimension

                padding_mask = padding_mask_from_original_shape(input_original_shape, target_shape)

                denoised_signal, _ =edm_sampler(
                                                    net, 
                                                    latents=padded_signal, 
                                                    sigma_max=input_sigma, 
                                                    padding_mask=padding_mask,
                                                    latents_already_noisy=True,
                                                    **sampler_kwargs
                                                )
                
                mask_slice = (slice(0, denoised_signal.shape[0]), )+tuple(slice(0, dim) for dim in original_shape[0])
                denoised_signal = denoised_signal[mask_slice]

                denoised_signal = denoised_signal * data_norm

                predict_SNR_result = cal_SNR(denoised_signal, true_signal, complex_axis=complex_axis)
                predict_SNR_result_sum_dist_list[idx] += torch.sum(predict_SNR_result)
        
        # End of dataloader
        if dist.get_rank() == 0:
            collect_predict_SNR_sum_list = [torch.zeros_like(predict_SNR_result_sum_dist_list, dtype=torch.float32, device=device) for _ in range(dist.get_world_size())]
        else:
            collect_predict_SNR_sum_list = None

        torch.distributed.gather(predict_SNR_result_sum_dist_list, gather_list=collect_predict_SNR_sum_list, dst=0)

        if dist.get_rank() == 0:
            # 모든 GPU에서 계산된 합계를 더함
            total_predict_SNR_sum = sum(collect_predict_SNR_sum_list)
            
            dist.print0("\n=== Denoising Test Results ===")
            for idx, snr_db in enumerate(SNR_steps):
                mean_snr_ratio = total_predict_SNR_sum[idx] / test_dataset_size
                predict_SNR_result_dict[snr_db] = ratio_to_dB(mean_snr_ratio.item())
                dist.print0(f"Input SNR: {snr_db:>5.1f} dB  ->  Mean Output SNR (Ratio): {predict_SNR_result_dict[snr_db]:>8.4f} (approx {ratio_to_dB(predict_SNR_result_dict[snr_db]):>7.4f} dB)")

        
        if dist.get_rank() == 0:
            import pandas as pd

            if opt.output:
                out_path = opt.output
                if out_path.endswith('.csv'):
                    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
                else:
                    os.makedirs(out_path, exist_ok=True)
                    out_path = os.path.join(out_path, 'predict_SNR_result.csv')
                
                pd.DataFrame(predict_SNR_result_dict, index=[0]).to_csv(out_path, index=False)
                dist.print0(f"Saved SNR results to {out_path}")

                opt_out_path = out_path.replace('.csv', '_opt.json')
                with open(opt_out_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(opt), f, indent=4)
                dist.print0(f"Saved options to {opt_out_path}")

        dist.destroy_process_group()

if __name__ == "__main__":
    main()