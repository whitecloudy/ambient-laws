import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T

# misc/ADJSCC 경로를 sys.path에 추가하여 Attention_Encoder 로드 지원
_CURRENT_DIR = Path(__file__).resolve().parent
_WORKSPACE_DIR = _CURRENT_DIR.parent
_ADJSCC_DIR = _WORKSPACE_DIR / 'misc' / 'ADJSCC'
if str(_ADJSCC_DIR) not in sys.path:
    sys.path.insert(0, str(_ADJSCC_DIR))

try:
    from util_module import Attention_Encoder
except ImportError:
    try:
        from misc.ADJSCC.util_module import Attention_Encoder
    except ImportError:
        Attention_Encoder = None

from training.dataset import renewRfProcessedDataset, parse_snr_range

logger = logging.getLogger(__name__)


def _load_adjscc_config(config_path_or_dict: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    JSON 또는 YAML 형식의 ADJSCC 설정 파일을 로드하거나 전달된 dict를 반환합니다.
    """
    if isinstance(config_path_or_dict, dict):
        return dict(config_path_or_dict)

    path = Path(config_path_or_dict)
    if not path.exists():
        raise FileNotFoundError(f"ADJSCC config file not found: {path}")

    ext = path.suffix.lower()
    if ext in ['.json']:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif ext in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML is required to parse YAML config files. Install via 'pip install pyyaml'.")
    else:
        # Default try JSON, fallback YAML
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}


class RenewSemanticCommunicationDataset(Dataset):
    """
    RENEW RF 채널 데이터와 ADJSCC를 결합하여 Semantic Communication (SC) 무선 전송을 모사하는 데이터셋 클래스.

    모사 과정:
    1) RENEW 데이터셋에서 RF 채널 데이터를 선택, RENEW의 corruption 메커니즘을 적용하고 해당 corruption 수준으로부터 SNR(dB)을 계산.
    2) 계산된 SNR 수준과 이미지 데이터셋에서 선택한 이미지를 ADJSCC의 Attention_Encoder에 입력하여 latent feature를 생성.
    3) 생성된 latent를 복소수 심볼로 변환 후 RF 채널 shape에 맞게 reshape 및 매칭.
       Clean 원본 채널에 latent를 곱하고 계산된 SNR 수준의 수신 노이즈를 더해 무선 전송을 최종 모사 (Y = H_clean * Z + Noise).
    4) 수신 신호 Y ('image')와 원본 clean RF 채널 정보 ('clean_channel')를 포함한 딕셔너리를 반환.
    """

    def __init__(
        self,
        adjscc_config: Union[str, Path, Dict[str, Any]],
        renew_dataset: Optional[renewRfProcessedDataset] = None,
        renew_path: Optional[str] = None,
        renew_kwargs: Optional[Dict[str, Any]] = None,
        image_dataset: Optional[Any] = None,
        image_random_seed: int = 42,
        matching_mode: str = 'interpolate',  # 'interpolate', 'pad_or_slice', 'repeat'
        power_normalize_latent: bool = True,
        **super_kwargs
    ):
        super().__init__()

        # 1. RENEW 데이터셋 초기화 또는 주입
        if renew_dataset is not None:
            self.renew_dataset = renew_dataset
        elif renew_path is not None:
            kwargs = dict(renew_kwargs or {})
            self.renew_dataset = renewRfProcessedDataset(path=renew_path, **kwargs)
        else:
            raise ValueError("Either 'renew_dataset' or 'renew_path' must be provided.")

        self.matching_mode = matching_mode
        self.power_normalize_latent = power_normalize_latent
        self.image_random_seed = int(image_random_seed)

        # 2. ADJSCC 설정 로드 및 Encoder 초기화
        self.adjscc_config = _load_adjscc_config(adjscc_config)
        self.tcn = int(self.adjscc_config.get('transmit_channel_num', 16))
        self.in_channels = int(self.adjscc_config.get('in_channels', 3))
        self.load_model_path = self.adjscc_config.get('load_model_path', None)
        self.device = torch.device(self.adjscc_config.get('device', 'cpu'))

        if Attention_Encoder is None:
            raise ImportError("Failed to import Attention_Encoder from misc/ADJSCC/util_module.py.")

        self.encoder = Attention_Encoder(tcn=self.tcn, in_channels=self.in_channels)

        if self.load_model_path is not None and str(self.load_model_path).strip() != '':
            model_path = Path(self.load_model_path)
            if model_path.exists():
                logger.info(f"Loading ADJSCC encoder weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'encoder' in checkpoint:
                    self.encoder.load_state_dict(checkpoint['encoder'])
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.encoder.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict):
                    # Filter keys for encoder if full model state_dict was saved
                    enc_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.')}
                    if enc_dict:
                        self.encoder.load_state_dict(enc_dict)
                    else:
                        self.encoder.load_state_dict(checkpoint, strict=False)
                else:
                    self.encoder.load_state_dict(checkpoint)
            else:
                logger.warning(f"ADJSCC model path {model_path} does not exist. Using randomly initialized encoder.")
        else:
            logger.info("No ADJSCC model checkpoint specified. Initialized Attention_Encoder with random weights.")

        self.encoder.to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 3. 이미지 데이터셋 초기화
        self.image_dataset = self._init_image_dataset(image_dataset)

    def _init_image_dataset(self, image_dataset: Optional[Any]):
        """이미지 데이터셋을 로드하거나 초기화합니다 (기본 CIFAR-10)."""
        if image_dataset is None:
            # CIFAR-10 로드 시도
            transform = T.Compose([
                T.ToTensor(),
            ])
            data_dir = os.path.join(_WORKSPACE_DIR, 'data', 'cifar10')
            try:
                from torchvision.datasets import CIFAR10
                cifar_dataset = CIFAR10(root=os.path.join(_WORKSPACE_DIR, 'data'), train=True, download=False, transform=transform)
                return cifar_dataset
            except Exception:
                # 다운로드 시도
                try:
                    from torchvision.datasets import CIFAR10
                    return CIFAR10(root=os.path.join(_WORKSPACE_DIR, 'data'), train=True, download=True, transform=transform)
                except Exception as e:
                    logger.warning(f"Failed to auto-download CIFAR-10 ({e}). Using a synthetic image dataset fallback.")
                    return DummyImageDataset(num_samples=1000, img_shape=(3, 32, 32))
        elif isinstance(image_dataset, (str, Path)):
            img_path = Path(image_dataset)
            if img_path.is_dir():
                from torchvision.datasets import ImageFolder
                return ImageFolder(str(img_path), transform=T.ToTensor())
            else:
                raise ValueError(f"Image dataset path {img_path} is not a valid directory.")
        elif isinstance(image_dataset, Dataset):
            return image_dataset
        else:
            raise TypeError(f"Unsupported image_dataset type: {type(image_dataset)}")

    def __len__(self) -> int:
        return len(self.renew_dataset)

    def _get_image(self, idx: int) -> Tuple[torch.Tensor, int]:
        """재현 가능한 랜덤 시드를 기반으로 이미지를 선택합니다."""
        img_rng = np.random.RandomState(int(self.image_random_seed + idx))
        img_idx = int(img_rng.randint(0, len(self.image_dataset)))
        item = self.image_dataset[img_idx]

        if isinstance(item, (tuple, list)):
            img, label = item[0], item[1]
        elif isinstance(item, dict):
            img = item.get('image', item.get('img'))
            label = item.get('label', 0)
        else:
            img, label = item, 0

        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)

        if img.ndim == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        img = img.float()
        return img, int(label) if isinstance(label, (int, np.integer)) else label

    def _match_latent_shape(self, z_complex: torch.Tensor, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        1D complex 심볼 텐서 z_complex를 target_shape의 numpy 복소수 배열로 매칭합니다.
        """
        target_numel = int(np.prod(target_shape))
        dim_z = z_complex.shape[0]

        if dim_z == target_numel:
            z_matched = z_complex.reshape(target_shape).cpu().numpy()
        elif self.matching_mode == 'interpolate':
            # 1D 선형 보간 (실수부 및 허수부 개별)
            real_1d = z_complex.real.unsqueeze(0).unsqueeze(0)  # (1, 1, dim_z)
            imag_1d = z_complex.imag.unsqueeze(0).unsqueeze(0)  # (1, 1, dim_z)

            real_interp = F.interpolate(real_1d, size=target_numel, mode='linear', align_corners=False).squeeze()
            imag_interp = F.interpolate(imag_1d, size=target_numel, mode='linear', align_corners=False).squeeze()

            z_interp = torch.complex(real_interp, imag_interp)
            z_matched = z_interp.reshape(target_shape).cpu().numpy()
        elif self.matching_mode == 'pad_or_slice':
            if dim_z < target_numel:
                pad_size = target_numel - dim_z
                padding = torch.zeros(pad_size, dtype=z_complex.dtype, device=z_complex.device)
                z_res = torch.cat([z_complex, padding], dim=0)
            else:
                z_res = z_complex[:target_numel]
            z_matched = z_res.reshape(target_shape).cpu().numpy()
        elif self.matching_mode == 'repeat':
            repeat_factor = int(np.ceil(target_numel / dim_z))
            z_rep = z_complex.repeat(repeat_factor)[:target_numel]
            z_matched = z_rep.reshape(target_shape).cpu().numpy()
        else:
            raise ValueError(f"Unknown matching_mode: {self.matching_mode}")

        # 단위 평균 전력 보정 (E[|z|^2] = 1)
        power_z = np.mean(np.abs(z_matched) ** 2)
        if power_z > 1e-12:
            z_matched = z_matched / np.sqrt(power_z)

        return z_matched

    def _format_complex_or_real(self, data: np.ndarray) -> np.ndarray:
        """view_as_complex 및 complex_merge_axis 설정에 맞춰 데이터를 포맷팅합니다."""
        if self.renew_dataset._view_as_complex:
            return data.astype(np.complex64)
        
        axis = self.renew_dataset._complex_merge_axis
        if axis is not None:
            res = np.append(data.real, data.imag, axis=axis)
        else:
            res = np.expand_dims(data, -1)
            res = np.append(data.real, data.imag, axis=-1)
        return res.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        item_fname = self.renew_dataset._fname[idx]

        # -------------------------------------------------------------
        # 1) RENEW RF 채널 데이터 로드 및 corruption 수행, SNR(dB) 산출
        # -------------------------------------------------------------
        csi_clean, noise_sigma_clean = self.renew_dataset._load_and_normalize(item_fname)
        csi_clean, noise_sigma_clean = self.renew_dataset._apply_transpose(csi_clean, noise_sigma_clean)
        # csi_clean는 복소수 numpy 배열 (frame, antenna, channel) 형태

        # Corruption 수행
        corr_item = {
            'image': csi_clean.copy(),
            'sigma': noise_sigma_clean.copy(),
            'idx': idx
        }
        corr_res = self.renew_dataset.corruption_collate_fn([corr_item])[0]
        csi_corrupted = corr_res['image']
        noise_sigma_corrupted = corr_res['sigma']

        # SNR(dB) 계산
        if self.renew_dataset.snr_range is not None:
            np_gen = np.random.default_rng(int(self.renew_dataset._image_corruption_seed + 175 + idx))
            snr_db = float(np_gen.uniform(self.renew_dataset.snr_range[0], self.renew_dataset.snr_range[1]))
        else:
            signal_power = float(np.mean(np.abs(csi_clean) ** 2))
            diff_noise = csi_corrupted - csi_clean
            noise_power = float(np.mean(np.abs(diff_noise) ** 2))
            if noise_power < 1e-12:
                noise_power = float(np.mean(noise_sigma_corrupted ** 2))

            if noise_power > 1e-12 and signal_power > 1e-12:
                snr_db = float(10.0 * np.log10(signal_power / noise_power))
            else:
                snr_db = 20.0  # 기본 SNR

        # -------------------------------------------------------------
        # 2) 이미지 선택 및 ADJSCC Encoder를 통한 latent 생성
        # -------------------------------------------------------------
        img_tensor, img_label = self._get_image(idx)
        img_batch = img_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        snr_batch = torch.tensor([[snr_db]], dtype=torch.float32, device=self.device)  # (1, 1)

        with torch.no_grad():
            features = self.encoder(img_batch, snr_batch)  # (1, tcn, H_out, W_out)

        # -------------------------------------------------------------
        # 3) Latent의 복소수 변환, reshape, 및 무선 채널 송수신 모사
        # -------------------------------------------------------------
        # ADJSCC 표준: 전반부=실수, 후반부=허수로 복소 심볼 생성
        f = features.reshape(1, -1)
        dim_z = f.shape[1] // 2
        z_in = torch.complex(f[:, :dim_z], f[:, dim_z:])  # (1, dim_z)

        if self.power_normalize_latent:
            norm_factor = torch.sum(z_in.real ** 2 + z_in.imag ** 2, dim=1, keepdim=True)
            scale = torch.sqrt(dim_z / torch.clamp(norm_factor, min=1e-8))
            z_in = z_in * torch.complex(scale, torch.zeros_like(scale))

        z_complex_1d = z_in.squeeze(0)  # (dim_z,)

        # RF 채널 shape에 맞게 매칭 (csi_clean.shape)
        z_matched = self._match_latent_shape(z_complex_1d, csi_clean.shape)

        # 무선 채널 송신 모사: Y_noiseless = H_clean * Z
        rx_noiseless = csi_clean * z_matched

        # SNR 수준의 수신단 노이즈 추가: Y = H_clean * Z + N
        rx_power = float(np.mean(np.abs(rx_noiseless) ** 2))
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_var = rx_power / max(snr_linear, 1e-12)
        noise_std = np.sqrt(noise_var)

        noise_rng = np.random.default_rng(int(self.renew_dataset._image_noise_seed + 8888 + idx))
        noise_real = noise_rng.standard_normal(csi_clean.shape)
        noise_imag = noise_rng.standard_normal(csi_clean.shape)
        rx_noise = (noise_real + 1j * noise_imag) / np.sqrt(2.0) * noise_std
        rx_signal = rx_noiseless + rx_noise

        # -------------------------------------------------------------
        # 4) view_as_complex 및 complex_merge_axis 포맷 적용 후 반환
        # -------------------------------------------------------------
        out_image = self._format_complex_or_real(rx_signal)
        out_clean_channel = self._format_complex_or_real(csi_clean)
        out_corrupted_channel = self._format_complex_or_real(csi_corrupted)



        return_dict = {
            'image': out_image,
            'clean_channel': out_clean_channel,
            'corrupted_channel': out_corrupted_channel,
            'snr_db': np.float32(snr_db),
            'latent': z_matched.astype(np.complex64),
            'source_image': img_tensor,
            'source_label': img_label,
            'sigma': noise_sigma_corrupted.astype(np.float32),
            'filename': item_fname,
            'idx': idx,
        }
        return return_dict


# 소문자 클래스명 별칭 지원 (renewRfProcessedDataset과 일관성 유지)
renewSemanticCommunicationDataset = RenewSemanticCommunicationDataset


class DummyImageDataset(Dataset):
    """오프라인 환경 또는 CIFAR-10이 없을 때 사용할 더미 이미지 데이터셋."""
    def __init__(self, num_samples: int = 1000, img_shape: Tuple[int, int, int] = (3, 32, 32)):
        self.num_samples = num_samples
        self.img_shape = img_shape

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rng = np.random.RandomState(idx + 1234)
        img = torch.from_numpy(rng.uniform(0.0, 1.0, size=self.img_shape)).float()
        label = int(idx % 10)
        return img, label
