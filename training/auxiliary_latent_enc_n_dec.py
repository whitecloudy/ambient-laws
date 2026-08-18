import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _init_weights(m):
    """Xavier (Glorot) Uniform initialization for Linear and Conv2d layers"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


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

        self.apply(_init_weights)

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

        self.apply(_init_weights)

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
        self.apply(_init_weights)

        self.log_m = math.log(m)

    def forward(self, x0, is_training=True):
        enc_dtype = next(self.encoder_backbone.parameters()).dtype
        x0 = x0.to(dtype=enc_dtype)
        # 1. UnetEncoder를 통한 Logits 추출
        logits = self.encoder_backbone(x0) # Shape: [B, m]
        
        if is_training:
            # 2. Sum-of-Gamma 노이즈 추가 (Sahoo et al.)
            # 원본 JAX 코드의 _gamma_noise 로직을 PyTorch로 이식
            shape = logits.shape
            gamma_tau = 10.0  # 논문 및 코드 기본값
        
            # alpha(concentration) = 1.0 / k 인 Gamma 분포 정의 (10번 샘플링)
            concentration = torch.full((10, *shape), 1.0 / self.k, dtype=logits.dtype, device=logits.device)
            gamma_dist = torch.distributions.Gamma(concentration, rate=1.0)
            noise = gamma_dist.sample() # Shape: [10, B, m]
            
            # beta 값 배열 생성: k / [1.0, 2.0, ..., 10.0]
            beta_vals = torch.arange(1.0, 11.0, dtype=logits.dtype, device=logits.device)
            beta = self.k / beta_vals
            
            # 차원 브로드캐스팅을 위해 view 사용 (예: [10, 1, 1] 형태로 변경)
            view_shape = [10] + [1] * len(shape)
            beta = beta.view(*view_shape)
            
            # Sum-of-Gamma 연산 적용
            s = noise / beta
            s = torch.sum(s, dim=0)          # 10번 샘플링한 차원을 기준으로 합산 -> Shape: [B, m]
            s = s - math.log(10.0)           # 상수 보정
            gamma_noise = gamma_tau * (s / self.k) # 최종 스케일링
            
            noisy_logits = logits + gamma_noise
        else:
            # 평가/추론 시에는 노이즈 생략
            noisy_logits = logits
            
        # 3. Top-K 하드 타겟 생성 (k-hot 벡터)
        _, topk_indices = torch.topk(noisy_logits, self.k, dim=-1)
        z_hard = torch.zeros_like(noisy_logits).scatter_(-1, topk_indices, 1.0)
        
        q = F.softmax(logits, dim=-1)
        log_q = F.log_softmax(logits, dim=-1)
        
        # 4. Identity Straight-Through Estimator (STE) 적용
        centered_noisy_logits = noisy_logits - noisy_logits.mean(dim=-1, keepdim=True)
        norm = torch.norm(centered_noisy_logits, p=2, dim=-1, keepdim=True)
        soft_topk = centered_noisy_logits / (norm + 1e-8)

        # 순전파: z_hard, 역전파: soft_topk의 그래디언트
        z = z_hard.detach() - soft_topk.detach() + soft_topk
        
        # 5. KL Divergence 계산 (Uniform prior 기준, 부동소수점 오차 방지를 위해 clamp min=0.0 적용)
        kl_loss = torch.clamp(torch.sum(q * (log_q + self.log_m), dim=-1), min=0.0).mean()
        
        return z, kl_loss

class noise_decoder(nn.Module):
    @property
    def output_len(self):
        if 'output_len' in self.__dict__:
            return self.__dict__['output_len']
        if '_output_len' in self.__dict__:
            return self.__dict__['_output_len']
        inout_dim = getattr(self, 'inout_dim', getattr(self, 'out_dim', [14, 8, 52]))
        res = 1
        for dim in inout_dim:
            res *= dim
        return res

    @output_len.setter
    def output_len(self, value):
        self.__dict__['output_len'] = value

    def __getstate__(self):
        state = super().__getstate__()
        state['scheduler_mode'] = getattr(self, 'scheduler_mode', 'no_nn_scheduler')
        state['output_len'] = self.output_len
        state['inout_dim'] = getattr(self, 'inout_dim', [14, 8, 52])
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if 'scheduler_mode' not in self.__dict__:
            self.scheduler_mode = 'no_nn_scheduler'
        if 'output_len' not in self.__dict__:
            inout_dim = getattr(self, 'inout_dim', getattr(self, 'out_dim', [14, 8, 52]))
            res = 1
            for d in inout_dim:
                res *= d
            self.output_len = res
        if 'inout_dim' not in self.__dict__:
            self.inout_dim = getattr(self, 'out_dim', [14, 8, 52])

    def __init__(self, m=50, inout_dim=[14,8,52], mlp_hidden_dim=None, cnn_dim=32, a_amp=4.0, b_amp=4.0, d_amp=2.0, scheduler_mode='using_sigma_t_n'):
        super().__init__()
        self.m = m
        self.inout_dim = inout_dim
        self.shape = (-1, ) + tuple(self.inout_dim) 
        self.cnn_dim = cnn_dim
        self.mlp_hidden_dim = 1
        self.scheduler_mode = scheduler_mode

        self.a_amp = a_amp
        self.b_amp = b_amp
        self.d_amp = d_amp

        if mlp_hidden_dim is None:
            for dim in self.inout_dim:
                self.mlp_hidden_dim *= dim
        else:
            self.mlp_hidden_dim = mlp_hidden_dim

        output_len = 1
        for dim in self.inout_dim:
            output_len *= dim
        self.output_len = output_len

        if self.scheduler_mode in ['using_sigma_t_n', 'using_sigma_t_n_wo_z']:
            m_encoder_output_len = cnn_dim//2 * inout_dim[1] * inout_dim[2]
            
            # 잠재 변수 z를 받아 다항식 계수를 생성하는 2-layer MLP
            self.m_encoder = nn.Sequential(
                nn.Linear(m, self.mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.mlp_hidden_dim, m_encoder_output_len),
                nn.LayerNorm(m_encoder_output_len)
            )

            self.sigma_t_n_encoder = nn.Sequential(
                nn.Conv2d(in_channels=self.inout_dim[0], out_channels=cnn_dim, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim//2, kernel_size=3, stride=1, padding=1),
                nn.LayerNorm((cnn_dim//2, inout_dim[1], inout_dim[2]))
            )

            self.output_noise_decoder = nn.Sequential(
                nn.Conv2d(in_channels=cnn_dim, out_channels=cnn_dim, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=cnn_dim, out_channels=int(inout_dim[0]*3), kernel_size=3, stride=1, padding=1) # a, b, d 세 가지 계수를 위해 3배수 출력
            )   
        elif self.scheduler_mode == 'no_sigma_t_n':
            # using_sigma_t_n이 False일 때: sigma_t_n을 사용하지 않고 MLP layer로 z만 다룸
            self.m_encoder = nn.Sequential(
                nn.Linear(m, self.mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.mlp_hidden_dim, self.output_len * 3)
            )
        elif self.scheduler_mode in ['no_nn_scheduler', 'no_nn_scheduler_with_z']:
            # z와 sigma_t_n을 사용하지 않고 a=0, b=0, d=1만 출력하는 모드
            pass
        else:
            raise ValueError(f"Unknown scheduler_mode: {self.scheduler_mode}")

        self.apply(_init_weights)

    def forward(self, z=None, sigma_t_n=None):
        scheduler_mode = getattr(self, 'scheduler_mode', 'no_nn_scheduler')
        if scheduler_mode in ['no_nn_scheduler', 'no_nn_scheduler_with_z']:
            if z is not None:
                batch_size = z.size(0)
                device = z.device
                dtype = z.dtype
            elif sigma_t_n is not None:
                batch_size = sigma_t_n.size(0)
                device = sigma_t_n.device
                dtype = sigma_t_n.dtype
            else:
                batch_size = 1
                device = torch.device('cpu')
                dtype = torch.float32

            a = torch.zeros((batch_size, self.output_len), device=device, dtype=dtype)
            b = torch.zeros((batch_size, self.output_len), device=device, dtype=dtype)
            d = torch.ones((batch_size, self.output_len), device=device, dtype=dtype)
            return a, b, d

        batch_size = z.size(0)
        dec_dtype = next(self.m_encoder.parameters()).dtype
        
        z = z.to(dtype=dec_dtype)

        if scheduler_mode in ['using_sigma_t_n', 'using_sigma_t_n_wo_z']:
            h, w = self.inout_dim[1], self.inout_dim[2]
            z_encoded = self.m_encoder(z)
            z_encoded = z_encoded.reshape(batch_size, self.cnn_dim // 2, h, w)

            if sigma_t_n.ndim < 4:
                while sigma_t_n.ndim < 4:
                    sigma_t_n = sigma_t_n.unsqueeze(-1)
                sigma_t_n = sigma_t_n.expand(batch_size, *self.inout_dim)

            sigma_t_n = sigma_t_n.to(dtype=dec_dtype)
            sigma_t_n_encoded = self.sigma_t_n_encoder(sigma_t_n)

            # latent code와 sigma_t_n_encoded를 컨캣하고 채널차원 기준으로 concatenation
            encoded = torch.cat([z_encoded, sigma_t_n_encoded], dim=1)

            noise_pred = self.output_noise_decoder(encoded)
            
            # c3 (inout_dim[0] * 3)를 3개로 분할하여 a, b, d 계수 추출
            a, b, d = torch.chunk(noise_pred, 3, dim=1)

            # PolynomialNoiseScheduler 다항식 계산에 맞춰 [B, output_len] 형태로 flatten
            a = (torch.sigmoid(a.reshape(batch_size, -1)) - 0.5) * self.a_amp
            b = (torch.sigmoid(b.reshape(batch_size, -1)) - 0.5) * self.b_amp
            d = (torch.sigmoid(d.reshape(batch_size, -1)) - 0.5) * self.d_amp + 1.0 
        elif scheduler_mode == 'no_sigma_t_n':
            noise_pred = self.m_encoder(z)
            a, b, d = torch.chunk(noise_pred, 3, dim=-1)

            a = (torch.sigmoid(a) - 0.5) * self.a_amp
            b = (torch.sigmoid(b) - 0.5) * self.b_amp
            d = (torch.sigmoid(d) - 0.5) * self.d_amp + 1.0
        else:
            raise ValueError(f"Unknown scheduler_mode: {scheduler_mode}")

        return a, b, d

        
class PolynomialNoiseScheduler(nn.Module):
    @property
    def scheduler_mode(self):
        return getattr(self, '_scheduler_mode', self.__dict__.get('scheduler_mode', 'no_nn_scheduler'))

    @scheduler_mode.setter
    def scheduler_mode(self, value):
        self._scheduler_mode = value

    def __getstate__(self):
        state = super().__getstate__()
        state['scheduler_mode'] = getattr(self, 'scheduler_mode', 'no_nn_scheduler')
        state['out_dim'] = getattr(self, 'out_dim', [14, 8, 52])
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if 'scheduler_mode' not in self.__dict__:
            self.scheduler_mode = 'no_nn_scheduler'
        if 'out_dim' not in self.__dict__:
            self.out_dim = [14, 8, 52]
    def __init__(self, m=50, out_dim=[14,8,52], sigma_max=80, sigma_min=0.002, rho=1, mlp_hidden_dim=None, cnn_dim=32, scheduler_mode='using_sigma_t_n'):
        super().__init__()
        self.m = m
        self.out_dim = out_dim
        self.shape = (-1, ) + tuple(out_dim) 
        output_len = 1

        for dim in out_dim:
            output_len *= dim

        if mlp_hidden_dim is None:
            mlp_hidden_dim = output_len

        self.scheduler_mode = scheduler_mode

        # noise_decoder 인스턴스 사용
        self.noise_decoder = noise_decoder(m=m, inout_dim=out_dim, mlp_hidden_dim=mlp_hidden_dim, cnn_dim=cnn_dim, scheduler_mode=self.scheduler_mode)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_max_rho = self.sigma_max ** (1 / self.rho)
        self.sigma_min_rho = self.sigma_min ** (1 / self.rho)

        self.apply(_init_weights)


    def __compute_tau_n(self, sigma_t_n):
        sigma_t_n_f64 = sigma_t_n.to(torch.float64)
        if sigma_t_n_f64.ndim > 1:
            sigma_t_n_mean = torch.sqrt(torch.mean(sigma_t_n_f64 ** 2, dim=list(range(1, sigma_t_n_f64.ndim))))
        else:
            sigma_t_n_mean = sigma_t_n_f64.abs()
        sigma_t_n_mean_rho = sigma_t_n_mean ** (1 / self.rho)

        tau_n = (self.sigma_max_rho - sigma_t_n_mean_rho) / (self.sigma_max_rho - self.sigma_min_rho)

        return tau_n

    def __compute_coefficients(self, z, sigma_t_n):
        a, b, d = self.noise_decoder(z, sigma_t_n) # 각각 [B, output_len]
        return a, b, d

    def __compute_f(self, a, b, d, t):
        a_f64 = a.to(torch.float64)
        b_f64 = b.to(torch.float64)
        d_f64 = d.to(torch.float64)
        t_f64 = t.to(torch.float64)
        f_t = ((a_f64**2 / 5) * t_f64**5 + \
                (a_f64 * b_f64 / 2) * t_f64**4 + \
                ((b_f64**2 + 2 * a_f64 * d_f64) / 3) * t_f64**3 + \
                (b_f64 * d_f64) * t_f64**2 + \
                (d_f64**2) * t_f64)
        return f_t  

    def __compute_sigma(self, abd, t, tau_n, sigma_t_n):
        a_f64 = abd[0].to(torch.float64)
        b_f64 = abd[1].to(torch.float64)
        d_f64 = abd[2].to(torch.float64)
        t_f64 = t.to(torch.float64)
        tau_n_f64 = tau_n.to(torch.float64)
        sigma_t_n_f64 = sigma_t_n.to(torch.float64)

        f_t = self.__compute_f(a_f64, b_f64, d_f64, t_f64)
        f_tau = self.__compute_f(a_f64, b_f64, d_f64, tau_n_f64)
        f_1 = self.__compute_f(a_f64, b_f64, d_f64, torch.tensor(1.0, dtype=torch.float64, device=t.device))

        sigma_t_n_rho = sigma_t_n_f64 ** (1 / self.rho)

        sigma_below = self.__compute_sigma_below_tau(f_t, f_tau, sigma_t_n_rho)
        sigma_above = self.__compute_sigma_above_tau(f_t, f_1, f_tau, sigma_t_n_rho)

        # Align dimensions for condition check
        t_aligned = t_f64
        while t_aligned.ndim < tau_n_f64.ndim:
            t_aligned = t_aligned.unsqueeze(-1)

        cond = t_aligned < tau_n_f64
        sigma = torch.where(cond, sigma_below, sigma_above)
        return sigma

    def __compute_sigma_below_tau(self, f_t, f_tau_n, sigma_t_n_rho):
        f_tau_n_safe = torch.clamp(f_tau_n.abs(), min=1e-8) * torch.sign(f_tau_n + 1e-12)
        base = self.sigma_max_rho + (f_t / f_tau_n_safe) * (sigma_t_n_rho - self.sigma_max_rho)
        return torch.clamp(base, min=1e-8) ** self.rho
    
    def __compute_sigma_above_tau(self, f_t, f_1, f_tau_n, sigma_t_n_rho):
        denom = f_tau_n - f_1
        denom_safe = torch.clamp(denom.abs(), min=1e-8) * torch.sign(denom + 1e-12)
        base = self.sigma_min_rho + (f_t - f_1) / denom_safe * (sigma_t_n_rho - self.sigma_min_rho)
        return torch.clamp(base, min=1e-8) ** self.rho

    def __compute_t_from_sigma(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma_tensor = torch.tensor(sigma, dtype=torch.float64)
        else:
            sigma_tensor = sigma.to(torch.float64)

        if sigma_tensor.ndim > 1:
            sigma_mean = torch.sqrt(torch.mean(sigma_tensor ** 2, dim=list(range(1, sigma_tensor.ndim))))
        else:
            sigma_mean = sigma_tensor

        sigma_mean_rho = sigma_mean ** (1 / self.rho)
        t = (self.sigma_max_rho - sigma_mean_rho) / (self.sigma_max_rho - self.sigma_min_rho)
        return t

    def forward(self, z, sigma, sigma_t_n, abd = None):
        """
        z: [B, m] (Encoder에서 나온 k-hot 벡터) 또는 None (no_nn_scheduler 모드인 경우)
        sigma: [B] 또는 스칼라/텐서 (Reference/Input Noise Level 값)
        sigma_t_n: [B, ...] (노이즈 타겟/상태 텐서)
        """
        B = sigma_t_n.size(0) if sigma_t_n is not None else (z.size(0) if z is not None else 1)
        orig_shape = sigma_t_n.shape

        # Ensure sigma and sigma_t_n are float64 tensors for high-power polynomial math
        if not isinstance(sigma, torch.Tensor):
            sigma_tensor = torch.tensor(sigma, device=sigma_t_n.device, dtype=torch.float64)
        else:
            sigma_tensor = sigma.to(device=sigma_t_n.device, dtype=torch.float64)

        sigma_t_n_f64 = sigma_t_n.to(torch.float64)
        sigma_expanded = sigma_tensor
        while sigma_expanded.ndim < sigma_t_n.ndim:
            sigma_expanded = sigma_expanded.unsqueeze(-1)
        sigma_expanded = sigma_expanded.expand(orig_shape)

        # 예외 처리 조건 확인: sigma > sigma_max 또는 sigma < sigma_min
        out_of_bounds = (sigma_expanded > self.sigma_max) | (sigma_expanded < self.sigma_min)

        # 전체가 범위 밖인 경우 MLP 및 다항식 계산 없이 즉시 반환
        if torch.all(out_of_bounds):
            return sigma_expanded, abd

        # sigma를 t (0~1 사이)로 환산 (float64)
        t = self.__compute_t_from_sigma(sigma_tensor)

        # Calculate tau_n (float64)
        tau_n = self.__compute_tau_n(sigma_t_n_f64)
        
        # Neural Network Architecture forward pass using module precision (e.g. bfloat16/float32)
        if abd is None:
            a, b, d = self.__compute_coefficients(z, sigma_t_n)
            abd = (a, b, d)
        else:
            a, b, d = abd
        
        a_f64 = a.to(torch.float64)
        b_f64 = b.to(torch.float64)
        d_f64 = d.to(torch.float64)
        abd_f64 = (a_f64, b_f64, d_f64)

        # 브로드캐스팅을 위해 t의 차원을 [B, 1]로 맞춤
        t_view = t.reshape(B, 1)
        t_view_clamped = torch.clamp(t_view, min=0.0, max=1.0)
        
        # Flatten inputs for polynomial calculations
        tau_n_flat = tau_n.reshape(B, 1)
        sigma_t_n_flat = sigma_t_n_f64.reshape(B, -1)
        
        # Compute polynomial sigma in float64
        sigma_flat = self.__compute_sigma(abd_f64, t_view_clamped, tau_n_flat, sigma_t_n_flat)
        target_shape = (B,) + tuple(self.out_dim) if len(orig_shape) == 1 else orig_shape
        polynomial_sigma = sigma_flat.reshape(target_shape)

        # 일부분만 범위 밖인 경우 torch.where 적용, 전부 범위 안이면 polynomial_sigma 즉시 반환
        if torch.any(out_of_bounds):
            return torch.where(out_of_bounds, sigma_expanded, polynomial_sigma), abd
        
        return polynomial_sigma, abd



