# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import torch.nn as nn
import dnnlib
import tqdm
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import ambient_utils
import wandb
import tempfile
from training.dataset import renewRfProcessedDataset, WiDARDataset, XRF55Dataset
import torch.multiprocessing as mp
import io
from validation.XRF55_validation import XRF55Validator


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


# -------------------------------------------------------------------
# [핵심 변경 사항 1] 저장 함수를 Global 영역으로 분리
# - multiprocessing은 타겟 함수를 pickle할 수 있어야 하므로 최상단에 위치해야 합니다.
# -------------------------------------------------------------------
def _mp_save_training_state(net_state, opt_state, nimg, cache_net, path, remove_path=None):
    # 1. 껍데기 복사
    cpu_net = copy.deepcopy(cache_net)
    
    # 2. 메인 프로세스(학습 루프)와 공유된 메모리 텐서들을 덮어씀
    cpu_net.load_state_dict(net_state)
    
    # 3. 디스크 직렬화 (이 작업은 완벽히 독립된 프로세스에서 실행되므로 GIL에 영향을 주지 않음)
    torch.save(dict(net=cpu_net, optimizer_state=opt_state, nimg=nimg), path)

    # 4. 저장이 완료된 후 이전 파일 삭제
    if remove_path is not None and os.path.exists(remove_path):
        try:
            os.remove(remove_path)
        except OSError:
            pass

def save_training_state(dir, net, optimizer, cur_nimg, remove_path=None):
    if dist.get_rank() != 0:
        return None

    unwrapped_net = net.module if hasattr(net, 'module') else net
    
    # -------------------------------------------------------------------
    # [핵심 변경 사항 2] CPU 복사본 생성 후 share_memory_() 호출
    # - 프로세스 간 통신(IPC) 시 텐서 데이터가 복사되는 병목을 원천 차단합니다.
    # -------------------------------------------------------------------
    net_state_cpu = {}
    for k, v in unwrapped_net.state_dict().items():
        t = v.detach().cpu().clone()
        t.share_memory_()  # 다른 프로세스에서 읽을 수 있도록 공유 메모리에 등록
        net_state_cpu[k] = t
    
    def _tensors_to_cpu_shared(obj):
        if torch.is_tensor(obj):
            t = obj.detach().cpu().clone()
            t.share_memory_()
            return t
        elif isinstance(obj, dict):
            return {k: _tensors_to_cpu_shared(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_tensors_to_cpu_shared(v) for v in obj]
        else:
            return obj
            
    opt_state_cpu = _tensors_to_cpu_shared(optimizer.state_dict())
    
    # -------------------------------------------------------------------
    # CPU 모델 껍데기 캐싱 (기존과 동일하게 최초 1회만 동작)
    # -------------------------------------------------------------------
    if not hasattr(save_training_state, "_cpu_net_cache"):
        temp_buf = io.BytesIO()
        torch.save(unwrapped_net, temp_buf)
        temp_buf.seek(0)
        save_training_state._cpu_net_cache = torch.load(temp_buf, map_location='cpu', weights_only=False)
        save_training_state._cpu_net_cache.eval().requires_grad_(False)
        temp_buf.close()

    file_path = os.path.join(dir, f'training-state-{cur_nimg//1000:06d}.pt')

    # -------------------------------------------------------------------
    # Background Process 생성 및 실행 (Thread 대신 Process 사용)
    # -------------------------------------------------------------------
    p = mp.Process(
        target=_mp_save_training_state, 
        args=(net_state_cpu, opt_state_cpu, cur_nimg, save_training_state._cpu_net_cache, file_path, remove_path)
    )
    p.start()

    return p

# -------------------------------------------------------------------
# [핵심 1] 프로세스 타겟 함수 최상단(Global) 분리
# -------------------------------------------------------------------
def _mp_save_network_snapshot(safe_data, tensor_states, cache, path, remove_path=None):
    # safe_data: GPU 객체가 배제된 순수 딕셔너리 (dataset_kwargs 등)
    final_dict = dict(safe_data)
    
    # 텐서 상태가 존재하는 키(nn.Module 객체들)에 대해서만 모델 재조립
    for k in tensor_states.keys():
        # 1. 순수 CPU 껍데기 모델 복사
        cpu_model = copy.deepcopy(cache[k])
        # 2. 공유 메모리로 넘어온 텐서 덮어쓰기
        cpu_model.load_state_dict(tensor_states[k])
        # 3. 최종 저장 딕셔너리에 삽입
        final_dict[k] = cpu_model
            
    with open(path, 'wb') as f:
        pickle.dump(final_dict, f)
    
    # 저장이 완료된 후 이전 파일 삭제
    if remove_path is not None and os.path.exists(remove_path):
        try:
            os.remove(remove_path)
        except OSError:
            pass



def save_network_snapshot(dir, ema, loss_fn, augment_pipe, dataset_kwargs, cur_nimg, check_ddp=True, remove_path=None):
    data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
    
    # -------------------------------------------------------------------
    # [핵심 2] GPU 텐서 추출 및 공유 메모리(Shared Memory) 등록
    # -------------------------------------------------------------------
    tensor_snapshots = {}
    safe_data = {}  # Multiprocessing IPC 통신을 위한 안전한 딕셔너리
    
    for key, value in data.items():
        if isinstance(value, torch.nn.Module):
            if check_ddp:
                misc.check_ddp_consistency(value)
            
            # 텐서를 CPU로 내리고 프로세스 간 공유 활성화 (복사 오버헤드 0)
            state_dict_shared = {}
            for k, v in value.state_dict().items():
                t = v.detach().cpu().clone()
                t.share_memory_()
                state_dict_shared[k] = t
                
            tensor_snapshots[key] = state_dict_shared
            
            # GPU 모델 본체는 프로세스 인자로 넘기면 안 되므로 None으로 처리
            safe_data[key] = None 
        else:
            # nn.Module이 아닌 일반 데이터(dict 등)는 그대로 전달
            safe_data[key] = value 
            
    torch.cuda.empty_cache()
    
    if dist.get_rank() != 0 or dir is None:
        return None

    # -------------------------------------------------------------------
    # [핵심 3] CPU 모델 껍데기 캐싱 (최초 1회만)
    # -------------------------------------------------------------------
    if not hasattr(save_network_snapshot, "_cpu_model_cache"):
        save_network_snapshot._cpu_model_cache = {}
        
    for key, value in data.items():
        if isinstance(value, torch.nn.Module) and key not in save_network_snapshot._cpu_model_cache:
            temp_buf = io.BytesIO()
            torch.save(value, temp_buf)
            temp_buf.seek(0)
            cpu_value = torch.load(temp_buf, map_location='cpu', weights_only=False)
            save_network_snapshot._cpu_model_cache[key] = cpu_value.eval().requires_grad_(False)
            temp_buf.close()

    file_path = os.path.join(dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')

    # -------------------------------------------------------------------
    # [핵심 4] 독립 프로세스(Process) 생성 및 직렬화 위임
    # -------------------------------------------------------------------
    # data 원본 대신 GPU 객체가 제거된 safe_data를 전달해야 에러가 나지 않습니다.
    p = mp.Process(
        target=_mp_save_network_snapshot, 
        args=(safe_data, tensor_snapshots, save_network_snapshot._cpu_model_cache, file_path, remove_path)
    )
    p.start()
    
    return p
#----------------------------------------------------------------------------


def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    validation_kwargs   = {},       # Options for validation, including whether to turn on validation, validation interval, validation iterations, validation batch size, and validation data.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    task                = 'RENEW',  # Current task
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    no_asm              = False,    # Force no ASM Loss
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    wandb_onoff         = False,    # Enable wandb logging
    allow_tf32          = False,
    debug_test          = False,
    temp_save           = False,   
    grad_clip           = 1000,
):
    # Initialize.
    start_time = time.time()
    nimg = None
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    if allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    if task == 'RENEW':
        dataset_obj = renewRfProcessedDataset(**dataset_kwargs)
    elif task == 'WIDAR':
        dataset_obj = WiDARDataset(**dataset_kwargs)
    elif task == 'XRF55':
        dataset_obj = XRF55Dataset(**dataset_kwargs)
    else:
        raise ValueError(f'Unsupported task: {task}')
    validation_on_off = validation_kwargs.validation_on_off
    if validation_on_off:
        validation_interval_tick = validation_kwargs.validation_interval
        
        if task != 'XRF55':
            validation_dataset_kwargs = dataset_kwargs.copy()
            if validation_kwargs.validation_data is not None:
                validation_dataset_kwargs['path'] = validation_kwargs.validation_data
                validation_dataset_kwargs['dataset_keep_percentage'] = 1.0
            else:
                validation_dataset_kwargs['flip_keep_dataset'] = True
            
            validation_dataset_obj = dataset_obj.__class__(**validation_dataset_kwargs)
            validation_dataset_sampler = misc.FiniteSampler(dataset=validation_dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=False, seed=seed)
            validation_dataset_iterator = torch.utils.data.DataLoader(dataset=validation_dataset_obj, sampler=validation_dataset_sampler, batch_size=validation_kwargs.validation_batch_size, **data_loader_kwargs)
        else:
            validation_dataset_iterator = None
            
            sampler_json_path = './misc/validation/sampler_kwargs.json'
            if os.path.exists(sampler_json_path):
                with open(sampler_json_path, 'r') as f:
                    sampler_kwargs = json.load(f)
            else:
                sampler_kwargs = {
                    "num_steps": 18,
                    "sigma_min": 0.002,
                    "sigma_max": 80.0,
                    "rho": 7.0,
                    "S_churn": 0.0,
                    "S_min": 0.0,
                    "S_noise": 1.0
                }
            
            xrf55_validator = XRF55Validator(device=device, **sampler_kwargs)
        
    else:
        validation_dataset_iterator = None
        validation_interval_tick = -1

    ## using This sampler is way way way~~~ too slow for every epoch renewal
    # dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=True, seed=seed)
    dist.print0('Dataset Loading completed...')
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dist.print0('Sampler Loading completed...')
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    dataset_item = next(dataset_iterator)
    dataset_shape = dataset_item["image"].shape
    dist.print0(f'Dataset image shape: {dataset_shape}')
    # Initialize temporary directory for training state dumps
    if dist.get_rank() == 0 and not debug_test and temp_save:
        run_dir_name = os.path.basename(os.path.normpath(run_dir))
        temp_dir_path = tempfile.mkdtemp(prefix='ambient-rf_'+run_dir_name+'_')
        latest_saved_kimg = None
        dist.print0(f'Temporary directory for training state dumps: {temp_dir_path}')
    else:
        temp_dir_path = None
        latest_saved_kimg = None
    
    # Construct network.
    dist.print0('Constructing network...')
    # interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    if task == 'RENEW':  
        interface_kwargs = dict(img_resolution=dataset_shape[-2:], img_channels=dataset_shape[-3], label_dim=dataset_obj.label_dim, label_resolution=dataset_shape[-2:])
    elif task in ['WIDAR', 'XRF55']:
        interface_kwargs = dict(img_resolution=dataset_shape[-2:], img_channels=dataset_shape[-3], label_dim=dataset_obj.label_dim, label_type='classes')
    else:
        interface_kwargs = dict(img_resolution=dataset_shape[-2:], img_channels=dataset_shape[-3], label_dim=dataset_obj.label_dim)

    # network_kwargs에 이미 label_dim이나 label_type이 명시되어 있으면(예: EDM scheduler 사용 시) interface_kwargs의 덮어쓰기 방지
    if 'label_dim' in network_kwargs:
        interface_kwargs.pop('label_dim', None)
    if 'label_type' in network_kwargs:
        interface_kwargs.pop('label_type', None)

    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    with torch.no_grad():
        images = torch.zeros(dataset_shape, device=device)
        sigma = torch.ones([batch_gpu], device=device)
        if hasattr(net, "generate_latent_z"):
            z, _ = net.generate_latent_z(images, sigma)
            labels = z
        elif getattr(getattr(net, 'model', None), 'label_type', None) != 'no_label' and "label" in dataset_item and dataset_item["label"] is not None:
            labels = torch.zeros_like(dataset_item["label"]).to(device)
        else:
            labels = None
        misc.print_module_summary(net, [images, sigma, labels], max_nesting=2, verbose=dist.get_rank() == 0)

    # Setup optimizer.
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    dist.print0('Setting up optimizer...')
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    # augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    augment_pipe = None
    dist.print0('Setting DDP...')
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=True, find_unused_parameters=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() == 0:
            with dnnlib.util.open_url(resume_pkl, verbose=True) as f:
                data = pickle.load(f)
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
            del data # conserve memory
        
        if dist.get_world_size() > 1:
            for param in misc.params_and_buffers(ema):
                torch.distributed.broadcast(param, src=0)
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        if 'nimg' in data:
            nimg = int(data['nimg'])
        del data # conserve memory
    torch.cuda.empty_cache()

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    mp.set_sharing_strategy('file_system')
    if nimg is not None:
        cur_nimg = nimg
    else:
        cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    bg_threads = []

    if debug_test:
        debug_start_time = time.time()

    # LOOP STARTS HERE.
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                dataset_item = next(dataset_iterator)
                images = dataset_item["image"].to(device)                
                labels = dataset_item["label"].to(device)
                current_sigma = dataset_item["sigma"].to(device)
                # additive_noise_sigma = dataset_item['additive_noise_sigma'].to(device)

                if "original_shape" in dataset_item:
                    original_shape = dataset_item["original_shape"].to(device)
                else:
                    original_shape = None

                if (loss_kwargs.class_name != 'training.loss.EDMLoss_boosted_sigma') and no_asm:
                    current_sigma = torch.zeros_like(current_sigma)

                loss_out = loss_fn(net=ddp, images=images, labels=labels, current_sigma=current_sigma, augment_pipe=augment_pipe, original_shape=original_shape)
                if len(loss_out) == 4:
                    loss, x0_pred, sigma, kl_loss = loss_out
                else:
                    loss, x0_pred, sigma = loss_out
                    kl_loss = None

                if kl_loss is not None:
                    kl_coeff = getattr(loss_fn, 'kl_coeff', 1.0)
                    if hasattr(loss_fn, 'module'):
                        kl_coeff = getattr(loss_fn.module, 'kl_coeff', kl_coeff)
                    pure_loss = (loss - kl_coeff * kl_loss.to(loss.dtype)).detach()
                    training_stats.report('Loss/loss', pure_loss)
                    training_stats.report('Loss/kl_loss', kl_loss.detach())
                else:
                    training_stats.report('Loss/loss', loss.detach())
                if debug_test:
                    with torch.no_grad():
                        debug_end_time = time.time()
                        dist.print0(f'loss: {torch.mean(loss).item()}, tick_time: {(debug_end_time - debug_start_time):.4f}s')
                        if kl_loss is not None:
                            dist.print0(f'kl_loss: {torch.mean(kl_loss).item()}')
                        debug_start_time = time.time()
                (loss).sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        with torch.no_grad():
            training_stats.report('Loss/grad_norm', grad_norm.item())
        if debug_test:
            with torch.no_grad():
                dist.print0(f'grad_norm: {grad_norm.item()}')

        optimizer.step()


        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        
        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        total_time_per_tick = tick_end_time - tick_start_time
        sec_per_kimg = (total_time_per_tick) / (cur_nimg - tick_start_nimg) * 1e3
        reamin_kimg = (total_kimg * 1000 - cur_nimg)/1000
        eta_seconds = sec_per_kimg * reamin_kimg

        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"eta time {dnnlib.util.format_time(training_stats.report0('ETA time', eta_seconds)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', total_time_per_tick):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', sec_per_kimg):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        if dist.get_rank() == 0 and wandb_onoff and wandb.run is not None:
            wandb.run.summary['Progress/tick'] = cur_tick
            wandb.run.summary['Progress/kimg'] = cur_nimg / 1e3
            wandb.run.summary['ETA time'] = eta_seconds
            wandb.run.summary['Estimated end date_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + eta_seconds))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Run validation.
        if validation_on_off and (validation_interval_tick > 0) and ((cur_tick) % validation_interval_tick == 0) and cur_tick!=0:
            ddp.eval()
            with torch.no_grad():
                if task == 'XRF55':
                    dist.print0("Running XRF55 validation (IS/FID)...")
                    # we make 50,000 samples to evalutate IS and FID
                    # divide the work across the distributed processes
                    num_samples = 50000
                    if debug_test:
                        num_samples = dist.get_world_size()*16
                    num_samples = int((num_samples // dist.get_world_size())) * dist.get_world_size()
                    
                    (is_mean, is_std), fid_val = xrf55_validator.validate(
                        net=ema, 
                        num_samples=num_samples, 
                        batch_size=validation_kwargs.validation_batch_size, 
                        real_loader=None,
                        image_shape=dataset_shape[1:],
                        data_norm=dataset_kwargs.normalize_value
                    )
                    
                    training_stats.report('Validation/IS_mean', is_mean)
                    training_stats.report('Validation/IS_std', is_std)
                    if fid_val is not None:
                        training_stats.report('Validation/FID', fid_val)
                        dist.print0(f"Validation IS: {is_mean:.4f} ± {is_std:.4f}, FID: {fid_val:.4f}")
                    else:
                        dist.print0(f"Validation IS: {is_mean:.4f} ± {is_std:.4f}")
                        
                    if wandb_onoff and wandb.run is not None:
                        wandb_data = {
                            'Validation/IS_mean': is_mean,
                            'Validation/IS_std': is_std
                        }
                        if fid_val is not None:
                            wandb_data['Validation/FID'] = fid_val
                        wandb.log(wandb_data, step=int(cur_nimg/1e3))
                elif validation_dataset_iterator is not None:
                    with tqdm.tqdm(total=validation_kwargs.validation_iterations * len(validation_dataset_iterator), desc="Validation", disable=not dist.get_rank() == 0) as pbar:
                        for val_iter in range(validation_kwargs.validation_iterations):
                            for val_dataset_item in validation_dataset_iterator:
                                val_images = val_dataset_item["image"].to(device)
                                val_labels = val_dataset_item["label"].to(device)
                                val_current_sigma = val_dataset_item["sigma"].to(device)

                                if "original_shape" in val_dataset_item:
                                    val_original_shape = val_dataset_item["original_shape"].to(device)
                                else:
                                    val_original_shape = None

                                val_loss_out = loss_fn(net=ema, images=val_images, labels=val_labels, current_sigma=val_current_sigma, augment_pipe=None, original_shape=val_original_shape)
                                if len(val_loss_out) == 4:
                                    val_loss, _, _, val_kl_loss = val_loss_out
                                else:
                                    val_loss, _, _ = val_loss_out
                                    val_kl_loss = None

                                if val_kl_loss is not None:
                                    kl_coeff = getattr(loss_fn, 'kl_coeff', 1.0)
                                    if hasattr(loss_fn, 'module'):
                                        kl_coeff = getattr(loss_fn.module, 'kl_coeff', kl_coeff)
                                    val_pure_loss = (val_loss - kl_coeff * val_kl_loss.to(val_loss.dtype)).detach()
                                    training_stats.report('Validation/loss', val_pure_loss)
                                    training_stats.report('Validation/kl_loss', val_kl_loss.detach())
                                else:
                                    training_stats.report('Validation/loss', val_loss.detach())
                                pbar.update(1)
                        pbar.close()
            ddp.train()

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            t = save_network_snapshot(run_dir, ema, loss_fn, augment_pipe, dataset_kwargs, cur_nimg)
            if t is not None: bg_threads.append(t)
            # data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            # for key, value in data.items():
            #     if isinstance(value, torch.nn.Module):
            #         value = copy.deepcopy(value).eval().requires_grad_(False)
            #         misc.check_ddp_consistency(value)
            #         data[key] = value.cpu()
            #     del value # conserve memory
            # if dist.get_rank() == 0:
            #     with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
            #         pickle.dump(data, f)
            # del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            t = save_training_state(run_dir, net, optimizer, cur_nimg)
            if t is not None: bg_threads.append(t)
        
        # Update logs.
        training_stats.default_collector.update()

        # 완료된 쓰레드는 리스트에서 제거하여 메모리 누수 방지
        finished_threads = [t for t in bg_threads if not t.is_alive()]
        bg_threads = [t for t in bg_threads if t not in finished_threads]
        for t in finished_threads:
            t.join()
            if hasattr(t, 'close'):
                t.close()

        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            # report to wandb
            for key, value in training_stats.default_collector.as_dict().items():
                if wandb_onoff:
                    wandb.log({key: value}, step=int(cur_nimg/1e3))
            stats_jsonl.flush()

            # Save a copy of the training state dump to the temporary directory
            if temp_dir_path is not None:
                training_state_remove_path = None
                network_snapshot_remove_path = None

                if latest_saved_kimg is not None:   # remove the previous dump
                    # 이전 파일 삭제 전, 디스크 쓰기가 진행 중이라면 대기하여 충돌 방지 (Back-pressure)
                    for t in bg_threads:
                        t.join()
                        if hasattr(t, 'close'):
                            t.close()
                    bg_threads.clear()

                    training_state_remove_path = os.path.join(temp_dir_path, f'training-state-{latest_saved_kimg//1000:06d}.pt')
                    network_snapshot_remove_path = os.path.join(temp_dir_path, f'network-snapshot-{latest_saved_kimg//1000:06d}.pkl')
                # save the new dump
                t1 = save_training_state(temp_dir_path, net, optimizer, cur_nimg, remove_path=training_state_remove_path)
                t2 = save_network_snapshot(temp_dir_path, ema, loss_fn, augment_pipe, dataset_kwargs, cur_nimg, check_ddp=False, remove_path=network_snapshot_remove_path)
                if t1 is not None: bg_threads.append(t1)
                if t2 is not None: bg_threads.append(t2)
                # data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
                # with open(os.path.join(temp_dir_path, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                #     pickle.dump(data, f)
            
                latest_saved_kimg = cur_nimg
        dist.synchronize()

        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
    # End of main training loop.
    
    # Cleanup all background threads before exiting to ensure all files are properly written and resources are released
    if dist.get_rank() == 0:
        if len(bg_threads) > 0:
            dist.print0('Waiting for background save threads to finish...')
            for t in bg_threads:
                t.join()
                if hasattr(t, 'close'):
                    t.close()

    # Remove the temporary directory and its contents if it exists
    if dist.get_rank() == 0 and temp_dir_path is not None and latest_saved_kimg is not None:
        os.remove(os.path.join(temp_dir_path, f'training-state-{latest_saved_kimg//1000:06d}.pt')) 
        os.remove(os.path.join(temp_dir_path, f'network-snapshot-{latest_saved_kimg//1000:06d}.pkl'))
        os.removedirs(temp_dir_path)    
    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
