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
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import ambient_utils
import wandb
import tempfile


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


#----------------------------------------------------------------------------


def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = True
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
    dataset_obj = ambient_utils.dataset_utils.GaussianNoiseAdditiveCorruptedImageFolderDataset(**dataset_kwargs)
    # random indices for dataset visualization
    indices = [476716, 801177, 208667, 84697, 708005, 481119, 882784, 314948, 241315, 900832, 937237, 522057, 844026, 1021191, 789191, 668501]
    indices = [index % len(dataset_obj) for index in indices]
    images_to_save = [torch.tensor(dataset_obj[i]['image']) for i in indices]
    if dist.get_rank() == 0:
        ambient_utils.save_images(torch.stack(images_to_save), os.path.join(run_dir, "dataset.png"), save_wandb=True)
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    
    # Initialize temporary directory for training state dumps
    if dist.get_rank() == 0:
        run_dir_name = os.path.basename(os.path.normpath(run_dir))
        temp_dir = tempfile.TemporaryDirectory(prefix=run_dir_name+'_')
        temp_dir_path = temp_dir.name
        latest_saved_kimg = None
        dist.print0(f'Temporary directory for training state dumps: {temp_dir_path}')
    
    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    with torch.no_grad():
        images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        sigma = torch.ones([batch_gpu], device=device)
        labels = torch.zeros([batch_gpu, net.label_dim], device=device)
        misc.print_module_summary(net, [images, sigma, labels], max_nesting=2, verbose=dist.get_rank() == 0)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
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
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                dataset_item = next(dataset_iterator)
                images = dataset_item["image"].to(device)                
                labels = dataset_item["label"].to(device)
                current_sigma = dataset_item["sigma"].to(device)
                loss, x0_pred, sigma = loss_fn(net=ddp, images=images, labels=labels, current_sigma=current_sigma, augment_pipe=augment_pipe)

                # every 500 steps save the images
                if cur_tick % 500 == 0 and dist.get_rank() == 0:
                    # ambient_utils.save_images(x0_pred, os.path.join(run_dir, f"images_{cur_tick}.png"), save_wandb=True)
                    save_images_with_sigmas(x0_pred, os.path.join(run_dir, f"images_{cur_tick}.png"), sigmas=sigma, save_wandb=True)
                
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

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
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            # report to wandb
            for key, value in training_stats.default_collector.as_dict().items():
                wandb.log({key: value}, step=cur_tick * snapshot_ticks)
            stats_jsonl.flush()

            # Save a copy of the training state dump to the temporary directory
            if latest_saved_kimg is not None:   # remove the previous dump
                os.remove(os.path.join(temp_dir_path, f'training-state-{latest_saved_kimg//1000:06d}.pt')) 
                os.remove(os.path.join(temp_dir_path, f'network-snapshot-{latest_saved_kimg//1000:06d}.pkl'))

            # save the new dump
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(temp_dir_path, f'training-state-{cur_nimg//1000:06d}.pt'))
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            with open(os.path.join(temp_dir_path, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                pickle.dump(data, f)
            
            latest_saved_kimg = cur_nimg
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    temp_dir.cleanup()  # Remove temporary directory for training state dumps on normal completion
    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
