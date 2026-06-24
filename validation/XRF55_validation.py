from validation.calculate_dml_is import DMLInceptionScoreCalculator
from training.sampler import inference_edm_sampler
import torch
import numpy as np
import torch_utils.distributed as dist

class XRF55Validator:
    """
    Validation class for XRF55 dataset.
    Generates synthetic samples using inference_edm_sampler, caches features/logits
    along with real validation samples, and computes IS and FID metrics.
    
    This process is completely gradient-free.
    """
    def __init__(self, device=None, calculator=None, stats_path='./misc/validation/real_wifi_stats.npz', **sampler_kwargs):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # If calculator is not provided, instantiate the default calculator
        if calculator is None:
            self.calculator = DMLInceptionScoreCalculator(device=self.device, stats_path=stats_path)
        else:
            self.calculator = calculator
            
        self.sampler_kwargs = sampler_kwargs

    @torch.no_grad()
    def validate(self, net, num_samples, batch_size=64, real_loader=None, image_shape=None, class_labels=None):
        """
        Runs the validation process without calculating gradients.
        
        Parameters:
        - net (nn.Module): The model to validate (e.g. training net).
        - num_samples (int): Total number of samples to generate and evaluate.
        - batch_size (int): Local batch size used for data generation.
        - real_loader (DataLoader, optional): DataLoader containing real validation samples.
        - image_shape (tuple, optional): Shape of the images (channels, length/height, width) if real_loader is None.
        - class_labels (torch.Tensor, optional): Predefined class labels if real_loader is None.
        
        Returns:
        - is_score (tuple): (mean_is, std_is) 
        - fid_score (float or None): FID score
        """
        # 1. Clear previous cached features
        self.calculator.clear()
        
        # 2. Determine sample requirements per process (multi-GPU DDP support)
        world_size = dist.get_world_size()
        local_num_samples = (num_samples + world_size - 1) // world_size
        
        # DDP 환경에서 각 GPU의 Rank에 맞추어 독립적인 Generator 시드 세팅
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(torch.initial_seed() % (2**32)) + dist.get_rank())

        # 3. Model setup (unwrap from DDP wrapper if needed)
        unwrapped_net = net.module if hasattr(net, 'module') else net
        unwrapped_net.eval()
        
        local_real_count = 0
        local_gen_count = 0
        
        # Scenario 1: Real loader is provided
        if real_loader is not None:
            loader_iter = iter(real_loader)
            while local_real_count < local_num_samples or local_gen_count < local_num_samples:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(real_loader)
                    batch = next(loader_iter)
                
                real_images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device) if 'label' in batch else None
                
                # Ingest real images
                if local_real_count < local_num_samples:
                    current_batch_size = min(real_images.size(0), local_num_samples - local_real_count)
                    self.calculator.feed_real(real_images[:current_batch_size], is_predictions=False)
                    local_real_count += current_batch_size
                
                # Generate and ingest synthetic images
                if local_gen_count < local_num_samples:
                    current_batch_size = min(real_images.size(0), local_num_samples - local_gen_count)
                    latent_shape = real_images.shape[1:]
                    latents = torch.randn(current_batch_size, *latent_shape, generator=generator, device=self.device)
                    batch_labels = labels[:current_batch_size] if labels is not None else None
                    
                    gen_images, _ = inference_edm_sampler(
                        unwrapped_net, 
                        latents, 
                        class_labels=batch_labels, 
                        **self.sampler_kwargs
                    )
                    
                    self.calculator.feed_gen(gen_images, is_predictions=False)
                    local_gen_count += current_batch_size
                    
        # Scenario 2: Real loader is not provided (rely on pre-calculated real stats)
        else:
            if image_shape is None:
                raise ValueError("Either real_loader or image_shape must be provided.")
                
            while local_gen_count < local_num_samples:
                current_batch_size = min(batch_size, local_num_samples - local_gen_count)
                latents = torch.randn(current_batch_size, *image_shape, generator=generator, device=self.device)
                
                batch_labels = None
                if class_labels is not None:
                    indices = torch.arange(local_gen_count, local_gen_count + current_batch_size) % class_labels.size(0)
                    batch_labels = class_labels[indices].to(self.device)
                    
                gen_images, _ = inference_edm_sampler(
                    unwrapped_net, 
                    latents, 
                    class_labels=batch_labels, 
                    **self.sampler_kwargs
                )
                
                self.calculator.feed_gen(gen_images, is_predictions=False)
                local_gen_count += current_batch_size
                
        # 4. Compute overall metrics (will automatically gather data across GPUs if under DDP)
        is_mean, is_std = self.calculator.compute_is(dataset_type='gen')
        
        try:
            fid_val = self.calculator.compute_fid()
        except ValueError as e:
            dist.print0(f"FID Calculation warning: {e}")
            fid_val = None
            
        return (is_mean, is_std), fid_val


if __name__ == '__main__':
    print("Running validation tests for XRF55_validation.py...")
    
    # 1. Create a dummy model conforming to sampler expectations
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sigma_min = 0.002
            self.sigma_max = 80.0
            
        def round_sigma(self, sigma):
            return sigma
            
        def forward(self, x, sigma, class_labels=None):
            # Identity function to represent perfect denoising for dummy testing
            return x.clone().to(torch.float32)
            
    # 2. Create a dummy dataset (channels=270, length=500 representing WiFi CSI raw data)
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 20
        def __getitem__(self, idx):
            return {
                'image': torch.randn(270, 500).to(torch.float32),
                'label': torch.randint(0, 55, (1,)).squeeze()
            }
            
    dataset = DummyDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    
    net = DummyNet()
    # Instantiate the validator with low steps for quick testing
    validator = XRF55Validator(num_steps=5)
    
    try:
        (is_mean, is_std), fid_val = validator.validate(
            net=net, 
            num_samples=10, 
            batch_size=4, 
            real_loader=loader
        )
        print("\n--- Test Results ---")
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        print(f"FID Score: {fid_val}")
    except Exception as e:
        print(f"Test failed with error: {e}")
