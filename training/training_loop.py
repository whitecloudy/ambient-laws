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
from training.dataset import renewRfProcessedDataset, WiDARDataset


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


def save_images_with_sigmas(images, image_path, sigmas=None, num_rows=None, num_cols=None, save_wandb=False, down_factor=None, wandb_down_factor=None):
    import math
    import PIL
    import PIL.Image
    from PIL import ImageDraw, ImageFont


    def find_closest_factors(number):
        sqrt_number = int(math.sqrt(number))
        
        n = sqrt_number
        m = number // n
        
        while n * m != number:
            n += 1
            m = number // n

        return m, n

    if num_rows is None and num_cols is None:
        num_rows = int(np.sqrt(images.shape[0]))    
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    elif num_rows is None and num_cols is not None:
        num_rows = int(np.ceil(images.shape[0] / num_cols))
    elif num_rows is not None and num_cols is None:
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    
    if num_rows * num_cols != images.shape[0]:
        num_rows, num_cols = find_closest_factors(images.shape[0])
    
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    image_size = images.shape[-2]
    
    # --- 텍스트 추가를 위한 로직 시작 ---

    has_text = sigmas is not None
    if has_text:
        # 텍스트 개수가 이미지 개수와 맞는지 확인
        assert len(sigmas) == images.shape[0], "sigma 배열의 길이는 이미지의 개수와 같아야 합니다. sigma : {}, images : {}".format(len(sigmas), images.shape[0])
        # sigmas가 tensor인 경우 numpy로 변환
        sigmas = sigmas.cpu().squeeze().numpy()
        # 텍스트를 표시할 추가 공간 정의
        text_area_width = 40  # 텍스트를 위한 가로 공간 (픽셀)
        text_padding = 5      # 이미지와 텍스트 사이의 여백
        font_size = 20         # 폰트 크기
        
        font = ImageFont.load_default()
            
        cell_width = image_size + text_area_width
    else:
        cell_width = image_size


    grid_image = PIL.Image.new('RGB', (num_cols * cell_width, num_rows * image_size), 'white')
    draw = ImageDraw.Draw(grid_image)

    # 각 위치에 이미지와 텍스트 배치
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index >= images.shape[0]:
                continue
            
            # 원본 이미지
            img = PIL.Image.fromarray(image_np[index])
            
            # 이미지를 붙여넣을 위치 계산
            paste_x = j * cell_width
            paste_y = i * image_size
            
            # 캔버스에 이미지 붙여넣기
            grid_image.paste(img, (paste_x, paste_y))
            
            # 텍스트가 있는 경우, 이미지 오른쪽에 텍스트 그리기
            if has_text:
                text_to_draw = f"{sigmas[index]:.2f}" # 소수점 2자리까지 표시
                
                # 텍스트 높이를 계산하여 세로 중앙에 위치시키기
                text_box = draw.textbbox((0, 0), text_to_draw, font=font)
                text_height = text_box[3] - text_box[1]
                
                text_x = paste_x + image_size + text_padding
                text_y = paste_y + (image_size - text_height) / 2
                
                draw.text((text_x, text_y), text_to_draw, fill="black", font=font)
    
    # --- 텍스트 추가 로직 종료 ---

    if down_factor is not None:
        grid_image = grid_image.resize((grid_image.size[0] // down_factor, grid_image.size[1] // down_factor))
    
    grid_image.save(image_path)

    if save_wandb:
        import wandb

    if save_wandb and wandb.run is not None:
        if wandb_down_factor is not None:
            # resize for speed
            grid_image = grid_image.resize((grid_image.size[0] // wandb_down_factor, grid_image.size[1] // wandb_down_factor))
        wandb.log({"images/" + image_path.split("/")[-1]: wandb.Image(grid_image)})


import threading

def save_training_state(dir, net, optimizer, cur_nimg):
    start_time = time.time()
    if dist.get_rank() != 0:
        return None
        
    # GPU 메모리(VRAM) 낭비 없이 텐서를 시스템 RAM(CPU)으로 바로 복사(격리)
    net_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    
    opt_state = {}
    for k, v in optimizer.state_dict().items():
        if isinstance(v, dict):
            opt_state[k] = {sub_k: (sub_v.detach().cpu().clone() if isinstance(sub_v, torch.Tensor) else sub_v) for sub_k, sub_v in v.items()}
        elif isinstance(v, torch.Tensor):
            opt_state[k] = v.detach().cpu().clone()
        else:
            opt_state[k] = v
            
    # 무거운 직렬화(torch.save) 및 파일 쓰기는 쓰레드에서 전담 처리
    def _save_to_disk(n_st, o_st, path):
        torch.save(dict(net=n_st, optimizer_state=o_st, nimg=cur_nimg), path)
        del n_st, o_st # conserve memory
            
    file_path = os.path.join(dir, f'training-state-{cur_nimg//1000:06d}.pt')
    t = threading.Thread(target=_save_to_disk, args=(net_state, opt_state, file_path))
    t.start()

    return t

def save_network_snapshot(dir, ema, loss_fn, augment_pipe, dataset_kwargs, cur_nimg, check_ddp=True):
    data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
    for key, value in data.items():
        if isinstance(value, torch.nn.Module):
            # deepcopy를 수행하므로 학습 루프의 모델 객체와는 완전히 분리됨
            value = copy.deepcopy(value).eval().requires_grad_(False)
            if check_ddp:
                misc.check_ddp_consistency(value)

            data[key] = value.cpu()
        
    if dist.get_rank() != 0 or dir is None:
        return None
    # 독립된 복사본(snapshot_data)을 쓰레드로 전달
    def _save(snapshot_data):
        for key, value in snapshot_data.items():
            if isinstance(value, torch.nn.Module):
                snapshot_data[key] = value.cpu()
        with open(os.path.join(dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
            pickle.dump(snapshot_data, f)
        del snapshot_data # conserve memory
            
    t = threading.Thread(target=_save, args=(data,))
    t.start()
    return t


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
    else:
        raise ValueError(f'Unsupported task: {task}')
    validation_on_off = validation_kwargs.validation_on_off
    if validation_on_off:
        validation_dataset_kwargs = dataset_kwargs.copy()
        if validation_kwargs.validation_data is not None:
            validation_dataset_kwargs['data'] = validation_kwargs.validation_data
            validation_dataset_kwargs['keep_percentage'] = 1.0
        else:
            validation_dataset_kwargs['flip_keep_dataset'] = True
        
        validation_dataset_obj = dataset_obj.__class__(**validation_dataset_kwargs)
        validation_dataset_sampler = misc.FiniteSampler(dataset=validation_dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=False, seed=seed)
        validation_dataset_iterator = torch.utils.data.DataLoader(dataset=validation_dataset_obj, sampler=validation_dataset_sampler, batch_size=validation_kwargs.validation_batch_size, **data_loader_kwargs)
        validation_interval_tick = validation_kwargs.validation_interval
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
    if dist.get_rank() == 0 and not debug_test:
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
        interface_kwargs = dict(img_resolution=dataset_shape[-2:], img_channels=dataset_shape[-3], label_dim=dataset_obj.label_dim, label_resolution=dataset_shape[-2:]) # TODO: This is very clumsy. Need to fix ASAP
    elif task == 'WIDAR':
        interface_kwargs = dict(img_resolution=dataset_shape[-2:], img_channels=dataset_shape[-3], label_dim=dataset_obj.label_dim, label_type='classes')
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    with torch.no_grad():
        images = torch.zeros(dataset_shape, device=device)
        sigma = torch.ones([batch_gpu], device=device)
        if net.model.label_type == 'downlink':
            labels = torch.zeros([batch_gpu, net.label_dim, net.img_resolution[0], net.img_resolution[1]], device=device)
        elif net.model.label_type == 'classes':
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
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
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        if 'nimg' in data:
            nimg = int(data['nimg'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
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

                loss, x0_pred, sigma = loss_fn(net=ddp, images=images, labels=labels, current_sigma=current_sigma, augment_pipe=augment_pipe, original_shape=original_shape)
                training_stats.report('Loss/loss', loss)
                if debug_test:
                    with torch.no_grad():
                        dist.print0(f'loss: {torch.mean(loss).item()}')
                (loss).sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
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
        if validation_on_off and validation_dataset_iterator is not None and (validation_interval_tick > 0) and ((cur_tick+1) % validation_interval_tick == 0):
            ddp.eval()
            with torch.no_grad():
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

                            val_loss, _, _ = loss_fn(net=ddp, images=val_images, labels=val_labels, current_sigma=val_current_sigma, augment_pipe=None, original_shape=val_original_shape)
                            
                            training_stats.report('Validation/loss', val_loss.clone().detach())
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
                if latest_saved_kimg is not None:   # remove the previous dump
                    # 이전 파일 삭제 전, 디스크 쓰기가 진행 중이라면 대기하여 충돌 방지 (Back-pressure)
                    for t in bg_threads:
                        t.join()
                    bg_threads.clear()

                    os.remove(os.path.join(temp_dir_path, f'training-state-{latest_saved_kimg//1000:06d}.pt')) 
                    os.remove(os.path.join(temp_dir_path, f'network-snapshot-{latest_saved_kimg//1000:06d}.pkl'))
                # save the new dump
                t1 = save_training_state(temp_dir_path, net, optimizer, cur_nimg)
                t2 = save_network_snapshot(temp_dir_path, ema, loss_fn, augment_pipe, dataset_kwargs, cur_nimg, check_ddp=False)
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

    # Remove the temporary directory and its contents if it exists
    if dist.get_rank() == 0 and temp_dir_path is not None and latest_saved_kimg is not None:
        os.remove(os.path.join(temp_dir_path, f'training-state-{latest_saved_kimg//1000:06d}.pt')) 
        os.remove(os.path.join(temp_dir_path, f'network-snapshot-{latest_saved_kimg//1000:06d}.pkl'))
        os.removedirs(temp_dir_path)    
    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
