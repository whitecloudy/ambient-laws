import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    """표준 Diffusion/VDM 모델에서 주로 사용하는 Swish(SiLU) 기반 ResNet 블록"""
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        
        if downsample:
            self.pool = nn.AvgPool2d(2)
            
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool2d(2) if downsample else nn.Identity()
            )

    def forward(self, x):
        h = self.act(self.conv1(x))
        if self.downsample:
            h = self.pool(h)
        h = self.act(self.conv2(h))
        return h + self.shortcut(x)

class UnetEncoder(nn.Module):
    """
    MuLAN 논문에 기술된 4개의 ResNet 블록을 사용하는 인코더 아키텍처.
    입력 이미지를 처리하여 m 차원의 logit 벡터를 출력합니다.
    """
    def __init__(self, in_channels=3, m=50, base_channels=128):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 4개의 ResNet 블록 시퀀스 (논문 구성과 동일)
        self.down_blocks = nn.Sequential(
            ResnetBlock(base_channels, base_channels),
            ResnetBlock(base_channels, base_channels * 2, downsample=True),
            ResnetBlock(base_channels * 2, base_channels * 2),
            ResnetBlock(base_channels * 2, base_channels * 4, downsample=True)
        )
        
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 중간 레이어에 Dropout (0.1) 적용 - MuLAN 기본 세팅
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(base_channels * 4, m)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down_blocks(x)
        x = self.act(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class TopKDiscreteEncoder(nn.Module):
    def __init__(self, in_channels=3, m=50, k=15, base_channels=128):
        """
        m: 전체 차원 수 (논문 디폴트: 50)
        k: 1이 되는 요소의 수 (논문 디폴트: 15)
        """
        super().__init__()
        self.m = m
        self.k = k
        
        # 백본으로 ldm.model_mulan_epsilon 스타일의 UnetEncoder 사용
        self.encoder_backbone = UnetEncoder(in_channels=in_channels, m=m, base_channels=base_channels)

    def forward(self, x0, is_training=True):
        # 1. UnetEncoder를 통한 Logits 추출
        logits = self.encoder_backbone(x0) # Shape: [B, m]
        
        if is_training:
            # 2. Sum-of-Gamma / Gumbel 노이즈 추가 (연속적 이완)
            U = torch.rand_like(logits)
            noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
            noisy_logits = logits + noise
        else:
            # 평가/추론 시에는 노이즈 생략
            noisy_logits = logits
            
        # 3. Top-K 하드 타겟 생성 (k-hot 벡터)
        _, topk_indices = torch.topk(noisy_logits, self.k, dim=-1)
        z_hard = torch.zeros_like(noisy_logits).scatter_(-1, topk_indices, 1.0)
        
        q = F.softmax(logits, dim=-1)
        
        # 4. Identity Straight-Through Estimator (STE) 적용
        # 순전파: z_hard, 역전파: q의 그래디언트
        z = z_hard.detach() - q.detach() + q
        
        # 5. KL Divergence 계산 (Uniform prior 기준)
        log_m = torch.log(torch.tensor(self.m, dtype=torch.float32, device=logits.device))
        kl_loss = -torch.sum(q * (torch.log(q + 1e-8) + log_m), dim=-1).mean()
        
        return z, kl_loss

class PolynomialNoiseScheduler(nn.Module):
    def __init__(self, m=50, out_dim=[14,8,52], sigma_max=80, sigma_min=0.002, rho=7, mlp_hidden_dim=None):
        super().__init__()
        self.m = m
        self.out_dim = out_dim
        self.shape = (-1, ) + tuple(out_dim) 
        output_len = 1

        for dim in out_dim:
            output_len *= dim

        if mlp_hidden_dim is None:
            mlp_hidden_dim = output_len
        
        # 잠재 변수 z를 받아 다항식 계수를 생성하는 2-layer MLP
        self.noise_decoder = nn.Sequential(
            nn.Linear(m, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, 3 * output_len) # a, b, d 세 가지 계수를 위해 3배수 출력
        )
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_max_rho = self.sigma_max ** (1 / self.rho)
        self.sigma_min_rho = self.sigma_min ** (1 / self.rho)


    def __compute_tau_n(self, sigma_t_n):
        sigma_t_n_mean = torch.sqrt(torch.mean(sigma_t_n ** 2, dim=list(range(1, sigma_t_n.ndim))))
        sigma_t_n_mean_rho = sigma_t_n_mean ** (1 / self.rho)

        tau_n = (self.sigma_max_rho - sigma_t_n_mean_rho) / (self.sigma_max_rho - self.sigma_min_rho)

        return tau_n

    def __compute_coefficients(self, z):
        coeffs = self.noise_decoder(z) # [B, 3 * output_len]
        a, b, d = torch.chunk(coeffs, 3, dim=-1) # 각각 [B, output_len]
        return a, b, d

    def __compute_f(self, a, b, d, t):
        f_t = (a**2 / 5) * t**5 + \
                (a * b / 2) * t**4 + \
                ((b**2 + 2 * a * d) / 3) * t**3 + \
                (b * d) * t**2 + \
                (d**2) * t
        return f_t  

    def __compute_sigma(self, abd, t, tau_n, sigma_t_n):
        f_t = self.__compute_f(abd[0], abd[1], abd[2], t)
        f_tau = self.__compute_f(abd[0], abd[1], abd[2], tau_n)
        f_1 = self.__compute_f(abd[0], abd[1], abd[2], 1.0)

        sigma_t_n_rho = sigma_t_n ** (1 / self.rho)

        sigma_below = self.__compute_sigma_below_tau(f_t, f_tau, sigma_t_n_rho)
        sigma_above = self.__compute_sigma_above_tau(f_t, f_1, f_tau, sigma_t_n_rho)

        # Align dimensions for condition check
        t_aligned = t
        if isinstance(t_aligned, torch.Tensor):
            while t_aligned.ndim < tau_n.ndim:
                t_aligned = t_aligned.unsqueeze(-1)

        cond = t_aligned < tau_n
        sigma = torch.where(cond, sigma_below, sigma_above)
        return sigma

    def __compute_sigma_below_tau(self, f_t, f_tau_n, sigma_t_n_rho):
        return (self.sigma_max_rho + (f_t / f_tau_n) * (sigma_t_n_rho - self.sigma_max_rho))**self.rho
    
    def __compute_sigma_above_tau(self, f_t, f_1, f_tau_n, sigma_t_n_rho):
        return (self.sigma_min_rho + (f_t - f_1) / (f_tau_n - f_1) * (sigma_t_n_rho - self.sigma_min_rho))**self.rho

    def __compute_t_from_sigma(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        else:
            sigma_tensor = sigma

        if sigma_tensor.ndim > 1:
            sigma_mean = torch.sqrt(torch.mean(sigma_tensor ** 2, dim=list(range(1, sigma_tensor.ndim))))
        else:
            sigma_mean = sigma_tensor

        sigma_mean_rho = sigma_mean ** (1 / self.rho)
        t = (self.sigma_max_rho - sigma_mean_rho) / (self.sigma_max_rho - self.sigma_min_rho)
        return t

    def forward(self, z, sigma, sigma_t_n):
        """
        z: [B, m] (Encoder에서 나온 k-hot 벡터)
        sigma: [B] 또는 스칼라/텐서 (Reference/Input Noise Level 값)
        sigma_t_n: [B, ...] (노이즈 타겟/상태 텐서)
        """
        B = z.size(0)
        orig_shape = sigma_t_n.shape

        # Ensure sigma is a tensor and expanded to orig_shape
        if not isinstance(sigma, torch.Tensor):
            sigma_tensor = torch.tensor(sigma, device=sigma_t_n.device, dtype=sigma_t_n.dtype)
        else:
            sigma_tensor = sigma.to(device=sigma_t_n.device, dtype=sigma_t_n.dtype)

        sigma_expanded = sigma_tensor
        while sigma_expanded.ndim < sigma_t_n.ndim:
            sigma_expanded = sigma_expanded.unsqueeze(-1)
        sigma_expanded = sigma_expanded.expand(orig_shape)

        # 예외 처리 조건 확인: sigma > sigma_max 또는 sigma < sigma_min
        out_of_bounds = (sigma_expanded > self.sigma_max) | (sigma_expanded < self.sigma_min)

        # 전체가 범위 밖인 경우 MLP 및 다항식 계산 없이 즉시 반환
        if torch.all(out_of_bounds):
            return sigma_expanded

        # sigma를 t (0~1 사이)로 환산
        t = self.__compute_t_from_sigma(sigma_tensor)

        # Calculate tau_n
        tau_n = self.__compute_tau_n(sigma_t_n)
        
        # MLP를 통과하여 a, b, d 계수 추출
        a, b, d = self.__compute_coefficients(z)
        
        # 브로드캐스팅을 위해 t의 차원을 [B, 1]로 맞춤
        t_view = t.view(B, 1)
        t_view_clamped = torch.clamp(t_view, min=0.0, max=1.0)
        
        # Flatten inputs for polynomial calculations
        tau_n_flat = tau_n.view(B, 1)
        sigma_t_n_flat = sigma_t_n.view(B, -1)
        
        # Compute polynomial sigma
        sigma_flat = self.__compute_sigma((a, b, d), t_view_clamped, tau_n_flat, sigma_t_n_flat)
        polynomial_sigma = sigma_flat.view(orig_shape)

        # 일부분만 범위 밖인 경우 torch.where 적용, 전부 범위 안이면 polynomial_sigma 즉시 반환
        if torch.any(out_of_bounds):
            return torch.where(out_of_bounds, sigma_expanded, polynomial_sigma)
        
        return polynomial_sigma



