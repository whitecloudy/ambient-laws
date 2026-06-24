import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 🚀 1D ResNet Model Architecture (Self-contained)
# ==========================================

def conv3x3(in_planes, out_planes, stride=1, group=1):
    """3x3 1D convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    """1x1 1D convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, group=group)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetLargeBert3(nn.Module):
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetLargeBert3, self).__init__()
        self.inplanes = 256
        self.conv1 = nn.Conv1d(inchannel, 256, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(1024, 1024, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024 * block.expansion, activity_num)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        output = self.avg_pool(c4)
        output_bert = output.view(output.size(0), -1)
        output = self.fc(output_bert)

        return output, output_bert


def resnet18_mutual():
    """Returns a ResNet 18 model for mutual learning (WiFi)."""
    return ResNetLargeBert3(BasicBlock, [2, 2, 2, 2])


# ==========================================
# 🔧 Helper Functions & Calculation Logic
# ==========================================

def check_weight_file(path):
    """Checks if default weight file exists; downloads it from Google Drive if not."""
    url_wifi = 'https://drive.google.com/file/d/1RM2wEE3AjOv0aYKnOcMJMdx7PmQp3XNM/view?usp=sharing'
    target_file = os.path.join(path, 'model0_params.pth')
    if not os.path.exists(target_file):
        print(f"Weights file not found at '{target_file}'. Downloading from Google Drive...")
        try:
            import gdown
            os.makedirs(path, exist_ok=True)
            gdown.download(url_wifi, target_file, quiet=False, fuzzy=True)
        except ImportError:
            print("gdown package is not installed. Please install it with 'pip install gdown' or download the weights manually.")
            print(f"Download URL: {url_wifi}")
    else:
        print('Weights file already exists.')


class DMLInceptionScoreCalculator:
    """
    A class to calculate the Inception Score (IS) and Fréchet Inception Distance (FID)
    using the DML WiFi ResNet1D model.
    
    Supports asynchronous/streaming ingestion where you feed batches of data continuously
    (e.g., from an external DataLoader loop), cache the extracted logits/latents on CPU/NumPy,
    and trigger computation at the end of the loop.
    """
    def __init__(self, model=None, device=None, weight_path='./result/params/model0_params.pth', stats_path='./result/real_wifi_stats.npz'):
        # 1. Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model
        self.weight_path = weight_path
        self.stats_path = stats_path
        
        # Ingestion cache (stored as lists of numpy arrays to avoid PyTorch GPU memory leaks)
        self.real_logits = []
        self.real_latents = []
        self.gen_logits = []
        self.gen_latents = []
        
        # Pre-calculated real statistics cache (for fast FID computation)
        self.real_mu = None
        self.real_sigma = None

        if self.stats_path is not None:
            self.load_real_statistics(self.stats_path)

    def clear(self, clear_stats=False):
        """Clears cached logits and latents. If clear_stats=True, also clears loaded real statistics."""
        self.real_logits.clear()
        self.real_latents.clear()
        self.gen_logits.clear()
        self.gen_latents.clear()
        if clear_stats:
            self.real_mu = None
            self.real_sigma = None
            print("Calculator cache and pre-calculated statistics cleared.")
        else:
            print("Calculator cache cleared (pre-calculated statistics preserved).")

    def load_default_model(self):
        """Loads and prepares the default model and pre-trained weights."""
        try:
            model = resnet18_mutual()
            if os.path.exists(self.weight_path):
                model.load_state_dict(torch.load(self.weight_path, map_location='cpu'))
                print(f"Loaded default weights from {self.weight_path}")
            else:
                check_weight_file(os.path.dirname(self.weight_path))
                if os.path.exists(self.weight_path):
                    model.load_state_dict(torch.load(self.weight_path, map_location='cpu'))
                    print(f"Downloaded and loaded default weights from {self.weight_path}")
                else:
                    raise FileNotFoundError(f"Default weight file '{self.weight_path}' could not be loaded.")
            self.model = model
        except Exception as e:
            raise RuntimeError(f"Failed to load the default model/weights: {e}")

    def _process_data(self, data, is_predictions=False):
        """
        Helper to format data and run inference to extract logits and latents.
        Returns:
            logits (np.ndarray): shape (B, 55)
            latents (np.ndarray): shape (B, 1024) or None if is_predictions=True
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            data = data.clone()
        else:
            raise TypeError("Input data must be a PyTorch Tensor or a NumPy Array.")
            
        data = data.to(torch.float32)
        
        if is_predictions:
            # Data is already predictions/logits
            return data.cpu().numpy(), None

        # Data is raw inputs, we need to run model inference
        if self.model is None:
            self.load_default_model()
            
        self.model = self.model.to(self.device)
        self.model.eval()

        if len(data.shape) == 2:
            # Add batch dimension if single sample (270, 500) -> (1, 270, 500)
            data = data.unsqueeze(0)

        prediction_list = []
        latent_list = []
        batch_size = 64
        n_samples = data.size(0)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_data = data[i:i+batch_size].to(self.device)
                model_outputs, model_vecs = self.model(batch_data)
                prediction_list.append(model_outputs.cpu().numpy())
                latent_list.append(model_vecs.cpu().numpy())

        logits = np.concatenate(prediction_list, axis=0)
        latents = np.concatenate(latent_list, axis=0)
        return logits, latents

    def feed_real(self, data, is_predictions=False, is_latent=False):
        """
        Feeds a batch of real data (ground truth).
        
        Parameters:
        - data (Union[torch.Tensor, np.ndarray]): Logits, raw inputs, or pre-extracted latents.
        - is_predictions (bool): Set to True if data consists of pre-computed logits.
        - is_latent (bool): Set to True if data consists of pre-computed latent features.
        """
        if is_latent:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            self.real_latents.append(data)
        else:
            logits, latents = self._process_data(data, is_predictions=is_predictions)
            self.real_logits.append(logits)
            if latents is not None:
                self.real_latents.append(latents)

    def feed_gen(self, data, is_predictions=False, is_latent=False):
        """
        Feeds a batch of generated/synthetic data.
        
        Parameters:
        - data (Union[torch.Tensor, np.ndarray]): Logits, raw inputs, or pre-extracted latents.
        - is_predictions (bool): Set to True if data consists of pre-computed logits.
        - is_latent (bool): Set to True if data consists of pre-computed latent features.
        """
        if is_latent:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            self.gen_latents.append(data)
        else:
            logits, latents = self._process_data(data, is_predictions=is_predictions)
            self.gen_logits.append(logits)
            if latents is not None:
                self.gen_latents.append(latents)

    def calculate_is_score(self, batch_inputs, n_splits=10):
        """
        Calculates Inception Score from logits/predictions.
        """
        if isinstance(batch_inputs, np.ndarray):
            batch_inputs = torch.from_numpy(batch_inputs)
        elif isinstance(batch_inputs, torch.Tensor):
            batch_inputs = batch_inputs.clone()
        else:
            raise TypeError("Input 'batch_inputs' must be a PyTorch Tensor or a NumPy Array.")

        batch_inputs = batch_inputs.to(self.device)
        preds = F.softmax(batch_inputs, dim=1)
        
        epsilon = 1e-16
        n_samples = preds.size(0)
        
        if n_samples < n_splits:
            raise ValueError(f"Number of samples ({n_samples}) must be greater than or equal to n_splits ({n_splits}).")
            
        split_scores = []
        step = n_samples // n_splits
        
        for k in range(n_splits):
            part_preds = preds[k * step : (k + 1) * step]
            p_y = torch.mean(part_preds, dim=0, keepdim=True)
            kl = torch.sum(part_preds * (torch.log(part_preds + epsilon) - torch.log(p_y + epsilon)), dim=1)
            split_score = torch.exp(torch.mean(kl)).item()
            split_scores.append(split_score)
            
        return float(np.mean(split_scores)), float(np.std(split_scores))

    def compute_is(self, dataset_type='gen', n_splits=10):
        """
        Computes Inception Score (IS) from all currently cached logits.
        
        Parameters:
        - dataset_type (str): 'gen' to calculate IS for generated data, 'real' for real data.
        - n_splits (int): Number of splits for evaluation.
        """
        cache = self.gen_logits if dataset_type == 'gen' else self.real_logits
        if not cache:
            raise ValueError(f"No cached logits for dataset type '{dataset_type}'. Call feed_{dataset_type} first.")
            
        all_logits = np.concatenate(cache, axis=0)
        return self.calculate_is_score(all_logits, n_splits=n_splits)

    def save_real_statistics(self, filepath):
        """
        Computes mean and covariance of currently cached real latent features
        and saves them to a .npz file.
        """
        if not self.real_latents:
            raise ValueError("No cached real latent features to save. Call feed_real first.")
            
        real_latents = np.concatenate(self.real_latents, axis=0)
        mu_real = np.mean(real_latents, axis=0)
        sigma_real = np.cov(real_latents, rowvar=False)
        
        # Save to .npz
        np.savez(filepath, mu=mu_real, sigma=sigma_real)
        print(f"Real statistics successfully saved to {filepath}")

    def load_real_statistics(self, filepath):
        """
        Loads pre-calculated mean and covariance of real latent features from a .npz file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Statistics file '{filepath}' not found.")
            
        data = np.load(filepath)
        if 'mu' not in data or 'sigma' not in data:
            raise KeyError("Loaded file must contain 'mu' and 'sigma' keys.")
            
        self.real_mu = data['mu']
        self.real_sigma = data['sigma']
        print(f"Pre-calculated real statistics successfully loaded from {filepath}")

    def compute_fid(self):
        """
        Computes Fréchet Inception Distance (FID) between all currently cached real and generated latents.
        Uses loaded real statistics if available, otherwise computes them from cached real latents.
        """
        # Get real data statistics
        if self.real_mu is not None and self.real_sigma is not None:
            mu_real = self.real_mu
            sigma_real = self.real_sigma
        else:
            if not self.real_latents:
                raise ValueError("No cached real latent features and no pre-calculated real statistics loaded. Call feed_real or load_real_statistics first.")
            real_latents = np.concatenate(self.real_latents, axis=0)
            mu_real = np.mean(real_latents, axis=0)
            sigma_real = np.cov(real_latents, rowvar=False)
            
        if not self.gen_latents:
            raise ValueError("No cached generated latent features. Call feed_gen first.")
            
        gen_latents = np.concatenate(self.gen_latents, axis=0)
        
        # Calculate mean and covariance for generated data
        mu_gen = np.mean(gen_latents, axis=0)
        sigma_gen = np.cov(gen_latents, rowvar=False)
        
        # Ensure they are at least 1D/2D arrays
        mu_real = np.atleast_1d(mu_real)
        mu_gen = np.atleast_1d(mu_gen)
        sigma_real = np.atleast_2d(sigma_real)
        sigma_gen = np.atleast_2d(sigma_gen)
        
        diff = mu_real - mu_gen
        
        import scipy.linalg
        covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        # Handle numerical instability
        eps = 1e-6
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_real.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_gen + offset))
            
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2.0 * np.trace(covmean)
        return float(fid)

    # ==========================
    # Backward-compatible API
    # ==========================

    def calculate(self, data, is_predictions=True, batch_size=64, n_splits=10):
        """Calculates Inception Score directly. (Backward compatibility wrapper)"""
        self.clear()
        self.feed_gen(data, is_predictions=is_predictions)
        res = self.compute_is(dataset_type='gen', n_splits=n_splits)
        self.clear()
        return res

    def calculate_fid(self, real_data, gen_data, is_latent=False, batch_size=64):
        """Calculates FID score directly. (Backward compatibility wrapper)"""
        self.clear()
        self.feed_real(real_data, is_latent=is_latent)
        self.feed_gen(gen_data, is_latent=is_latent)
        res = self.compute_fid()
        self.clear()
        return res


def calculate_dml_is(data, is_predictions=True, model=None, device=None, batch_size=64, n_splits=10):
    """
    Calculates Inception Score (IS) from either inference-completed data or raw input data.
    Backward-compatible wrapper function for DMLInceptionScoreCalculator.
    """
    calculator = DMLInceptionScoreCalculator(model=model, device=device)
    return calculator.calculate(data, is_predictions=is_predictions, batch_size=batch_size, n_splits=n_splits)


if __name__ == '__main__':
    print("Running validation tests for self-contained calculate_dml_is.py (Class version with FID)...")
    
    # Instantiate the calculator class
    calculator = DMLInceptionScoreCalculator()
    
    # Test 1: Testing direct Inception Score calculation from predictions/logits (NumPy)
    print("\n--- Test 1: Predictions (NumPy array via Class) ---")
    np.random.seed(42)
    dummy_preds_np = np.random.randn(100, 10)  # 100 samples, 10 classes
    mean_score_np, std_score_np = calculator.calculate(dummy_preds_np, is_predictions=True, n_splits=10)
    print(f"Result (NumPy): {mean_score_np:.4f} ± {std_score_np:.4f}")
    
    # Test 2: Testing direct Inception Score calculation from predictions/logits (PyTorch Tensor)
    print("\n--- Test 2: Predictions (PyTorch Tensor via Class) ---")
    torch.manual_seed(42)
    dummy_preds_torch = torch.randn(100, 10)  # 100 samples, 10 classes
    mean_score_torch, std_score_torch = calculator.calculate(dummy_preds_torch, is_predictions=True, n_splits=10)
    print(f"Result (Tensor): {mean_score_torch:.4f} ± {std_score_torch:.4f}")
    
    # Test 3: Raw CSI input test
    print("\n--- Test 3: Raw Inputs Inference & Score via Class ---")
    try:
        # Dummy CSI inputs: (N, 270, 500)
        dummy_raw_inputs = np.random.rand(20, 270, 500).astype(np.float32)
        mean_score_raw, std_score_raw = calculator.calculate(
            dummy_raw_inputs, 
            is_predictions=False, 
            batch_size=10, 
            n_splits=2  # Since we only have 20 samples, use n_splits=2
        )
        print(f"Result (Raw Inputs): {mean_score_raw:.4f} ± {std_score_raw:.4f}")
    except Exception as e:
        print(f"Raw inputs test failed or weight file missing: {e}")

    # Test 4: Verify wrapper compatibility
    print("\n--- Test 4: Verification of backward-compatible wrapper function ---")
    mean_wrapper, std_wrapper = calculate_dml_is(dummy_preds_np, is_predictions=True, n_splits=10)
    print(f"Result (Wrapper): {mean_wrapper:.4f} ± {std_wrapper:.4f}")

    # Test 5: FID Score test using dummy features
    print("\n--- Test 5: FID Score calculation (Class) ---")
    dummy_real = np.random.randn(50, 1024)
    dummy_gen = np.random.randn(50, 1024) + 0.5  # shift slightly
    fid = calculator.calculate_fid(dummy_real, dummy_gen, is_latent=True)
    print(f"FID Score (latent): {fid:.4f}")

    # Test 6: FID Score test using raw inputs
    print("\n--- Test 6: FID Score calculation from raw inputs (Class) ---")
    try:
        dummy_real_raw = np.random.rand(20, 270, 500).astype(np.float32)
        dummy_gen_raw = np.random.rand(20, 270, 500).astype(np.float32)
        fid_raw = calculator.calculate_fid(
            dummy_real_raw, 
            dummy_gen_raw, 
            is_latent=False, 
            batch_size=10
        )
        print(f"FID Score (raw inputs): {fid_raw:.4f}")
    except Exception as e:
        print(f"Raw inputs FID test failed or weight file missing: {e}")

    # Test 7: Streaming / Feed API test (Asynchronous calculation)
    print("\n--- Test 7: Streaming / Feed API (Asynchronous) ---")
    calculator.clear(clear_stats=True)
    try:
        # Simulate batch feeding from a dataloader loop (5 batches)
        for i in range(5):
            batch_real = np.random.rand(4, 270, 500).astype(np.float32)
            batch_gen = np.random.rand(4, 270, 500).astype(np.float32)
            calculator.feed_real(batch_real, is_predictions=False)
            calculator.feed_gen(batch_gen, is_predictions=False)
            print(f"Fed batch {i+1}/5 to calculator.")
            
        # Now compute the scores
        is_mean, is_std = calculator.compute_is(dataset_type='gen', n_splits=2)
        fid_val = calculator.compute_fid()
        
        print(f"Result (Streaming IS): {is_mean:.4f} ± {is_std:.4f}")
        print(f"Result (Streaming FID): {fid_val:.4f}")
    except Exception as e:
        print(f"Streaming test failed: {e}")

    # Test 8: Save/Load Real Statistics for FID
    print("\n--- Test 8: Save/Load Real Statistics for FID ---")
    calculator.clear(clear_stats=True)
    try:
        # Feed real data to calculator
        dummy_real_data = np.random.randn(30, 1024)
        calculator.feed_real(dummy_real_data, is_latent=True)
        
        # Save statistics to a temp file in the workspace
        stats_path = './result/real_stats.npz'
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        calculator.save_real_statistics(stats_path)
        
        # Clear calculator including cached real features
        calculator.clear(clear_stats=True)
        
        # Load the saved statistics
        calculator.load_real_statistics(stats_path)
        
        # Feed new generated features
        dummy_gen_data = np.random.randn(30, 1024) + 0.5
        calculator.feed_gen(dummy_gen_data, is_latent=True)
        
        # Compute FID using the loaded statistics
        fid_loaded = calculator.compute_fid()
        print(f"Result (FID with loaded statistics): {fid_loaded:.4f}")
        
        # Clean up temp file
        if os.path.exists(stats_path):
            os.remove(stats_path)
    except Exception as e:
        print(f"Save/Load statistics test failed: {e}")


