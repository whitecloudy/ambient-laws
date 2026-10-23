import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

from training.dataset import (
    renewRfProcessedDataset,
    parse_snr_range,
    find_npz_files,
    CorruptionCollateFn,
    AlternativeCorruptionCollateFn,
    NoiseMeanCollateFn,
    sigma_mean,
)

logger = logging.getLogger(__name__)

# WiFi 802.11 OFDM 파일럿 서브캐리어 인덱스 (-21, -7, 7, 21 번에 해당)
# 52개 서브캐리어 (-26 ~ +26, DC 0 제외) 기준 인덱스 5, 19, 32, 46
PILOT_SUBCARRIER_INDICES_52 = (5, 19, 32, 46)
DATA_SUBCARRIER_INDICES_48 = tuple(i for i in range(52) if i not in PILOT_SUBCARRIER_INDICES_52)


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
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}


def calculate_sc_t_from_config(
    adjscc_config: Union[str, Path, Dict[str, Any]],
    image_shape: Optional[Tuple[int, int]] = None,
    num_subcarriers: int = 48
) -> int:
    """
    adjscc_config에 정의된 이미지 크기(image_size)와 압축 수준(tcn 또는 bandwidth ratio)을 이용하여
    Semantic Communication 전송에 필요한 시간 스텝 T를 자동으로 계산합니다.

    계산 방식:
      1) 이미지 해상도 (H, W):
         - adjscc_config의 'image_size' 또는 'img_size' (예: [32, 32] 또는 32)
         - 없을 경우 image_shape 참조 (예: (32, 32))
         - 모두 없을 경우 기본값 32 x 32 (CIFAR-10 기준)
      2) 압축 수준 (tcn / compression_ratio):
         - Attention_Encoder는 2단계의 stride-2 컨볼루션을 통해 H, W를 각각 1/4로 다운샘플링:
           H_out = H // 4, W_out = W // 4
         - 출력 채널 수 tcn (transmit_channel_num, 기본값 16)
         - 인코더 출력 실수(real) 심볼 수: k_real = tcn * H_out * W_out
         - 무선 전송용 복소수(complex) 심볼 수: k_complex = k_real // 2
           (만약 compression_ratio나 channel_bandwidth_ratio R이 주어지면 k_complex = ceil(R * 3 * H * W))
      3) 시간 스텝 T:
         - 48개 서브캐리어(SISO 단일 안테나) 기준:
           T = ceil(k_complex / 48)
    """
    if isinstance(adjscc_config, (str, Path)):
        adjscc_config = _load_adjscc_config(adjscc_config)

    # 1. 이미지 크기 결정 (H, W)
    img_size = adjscc_config.get('image_size', adjscc_config.get('img_size', None))
    if img_size is not None:
        if isinstance(img_size, (int, float)):
            h, w = int(img_size), int(img_size)
        elif len(img_size) == 2:
            h, w = int(img_size[0]), int(img_size[1])
        elif len(img_size) >= 3:
            h, w = int(img_size[-2]), int(img_size[-1])
        else:
            h, w = 32, 32
    elif image_shape is not None:
        h, w = int(image_shape[-2]), int(image_shape[-1])
    else:
        h, w = 32, 32

    # 2. 압축 수준 결정 (k_complex)
    h_out, w_out = max(1, h // 4), max(1, w // 4)
    if 'compression_ratio' in adjscc_config:
        ratio = float(adjscc_config['compression_ratio'])
        k_complex = int(np.ceil(ratio * (3 * h * w)))
    elif 'channel_bandwidth_ratio' in adjscc_config:
        ratio = float(adjscc_config['channel_bandwidth_ratio'])
        k_complex = int(np.ceil(ratio * (3 * h * w)))
    else:
        tcn = int(adjscc_config.get('transmit_channel_num', adjscc_config.get('tcn', 16)))
        k_real = tcn * h_out * w_out
        k_complex = max(1, k_real // 2)

    # 3. 48개 서브캐리어 기준 시간 스텝 T 계산
    t = int(np.ceil(k_complex / float(num_subcarriers)))
    return max(1, t)


def find_adjscc_checkpoint(
    adjscc_config: Dict[str, Any],
    explicit_path: Optional[Union[str, Path]] = None,
    config_file_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    ADJSCC 인코더 체크포인트(.pt) 경로를 찾습니다.
    우선순위:
    1) explicit_path (CLI 옵션 --adjscc_model_path 또는 생성자 인자)
    2) adjscc_config 내 'load_model_path' 또는 'model_path' (null/None이 아닌 경우)
    3) 자동 탐색 (ADJSCC 학습용 yaml 등에서 load_model_path: null 인 경우):
       config의 model_dir(예: misc/ADJSCC/model/)에서
       channel_type, tcn, snr, bs, lr 등의 하이퍼파라미터 패턴과 일치하는 .pt 모델 파일을 자동 탐색하여 반환.
    """
    # 1. explicit_path
    if explicit_path is not None and str(explicit_path).strip() not in ('', 'None', 'null'):
        cand = Path(explicit_path)
        if cand.exists():
            return cand.resolve()
        cand_cwd = (Path.cwd() / explicit_path).resolve()
        if cand_cwd.exists():
            return cand_cwd

    # 2. config['load_model_path'] / config['model_path']
    cfg_model_path = adjscc_config.get('load_model_path', adjscc_config.get('model_path', None))
    if cfg_model_path is not None and str(cfg_model_path).strip() not in ('', 'None', 'null'):
        cand = Path(cfg_model_path)
        if cand.exists():
            return cand.resolve()
        cand_cwd = (Path.cwd() / cfg_model_path).resolve()
        if cand_cwd.exists():
            return cand_cwd
        if config_file_path is not None:
            cand_cfg = (Path(config_file_path).resolve().parent / cfg_model_path).resolve()
            if cand_cfg.exists():
                return cand_cfg

    # 3. 자동 탐색 (ADJSCC 파일명 네이밍 규칙에 따른 검색)
    model_dir_str = adjscc_config.get('model_dir', 'misc/ADJSCC/model/')
    dir_candidates = [
        Path(model_dir_str),
        Path.cwd() / model_dir_str,
    ]
    if config_file_path is not None:
        cfg_dir = Path(config_file_path).resolve().parent
        dir_candidates.append(cfg_dir / model_dir_str)
        dir_candidates.append(cfg_dir / "model")

    valid_dir = None
    for d in dir_candidates:
        if d.is_dir():
            valid_dir = d.resolve()
            break

    if valid_dir is None:
        return None

    channel_type = str(adjscc_config.get('channel_type', '*')).lower().strip()
    tcn = adjscc_config.get('transmit_channel_num', adjscc_config.get('tcn', '*'))
    snr_low_train = adjscc_config.get('snr_low_train', None)
    snr_up_train = adjscc_config.get('snr_up_train', None)
    snr_low_eval = adjscc_config.get('snr_low_eval', None)
    snr_up_eval = adjscc_config.get('snr_up_eval', None)
    bs = adjscc_config.get('batch_size', None)
    lr = adjscc_config.get('learning_rate', None)

    glob_patterns = []
    # 1) exact match with train snr, bs, lr
    if snr_low_train is not None and snr_up_train is not None and bs is not None and lr is not None:
        glob_patterns.append(f"*_{channel_type}_tcn{tcn}_snrdb{snr_low_train}to{snr_up_train}_bs{bs}_lr{lr}.pt")
        glob_patterns.append(f"*_tcn{tcn}_snrdb{snr_low_train}to{snr_up_train}_bs{bs}_lr{lr}.pt")
    # 2) match with eval snr
    if snr_low_eval is not None and snr_up_eval is not None and bs is not None and lr is not None:
        glob_patterns.append(f"*_{channel_type}_tcn{tcn}_snrdb{snr_low_eval}to{snr_up_eval}_bs{bs}_lr{lr}.pt")
    # 3) match channel_type and tcn
    glob_patterns.append(f"*_{channel_type}_tcn{tcn}*.pt")
    # 4) match tcn
    glob_patterns.append(f"*tcn{tcn}*.pt")
    # 5) any pt file in model_dir
    glob_patterns.append("*.pt")

    for pat in glob_patterns:
        matched = sorted(list(valid_dir.glob(pat)))
        if matched:
            return matched[0].resolve()

    return None


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


def init_image_dataset(image_dataset: Optional[Any] = None) -> Dataset:
    """이미지 데이터셋을 로드하거나 초기화합니다 (기본 CIFAR-10)."""
    if image_dataset is None:
        transform = T.Compose([T.ToTensor()])
        cifar_candidates = [
            os.path.join(_WORKSPACE_DIR, 'data', 'cifar10', 'train'),
            './data/cifar10/train',
            '/data/alex9395/cifar10/train',
            os.path.join(str(_ADJSCC_DIR), 'data', 'cifar10', 'train'),
        ]
        for cand in cifar_candidates:
            if os.path.exists(cand) and os.path.isdir(cand):
                from torchvision.datasets import ImageFolder
                logger.info(f"Using existing CIFAR-10 dataset from {cand} (50,000 images found, no download needed)")
                return ImageFolder(cand, transform=transform)

        try:
            from torchvision.datasets import CIFAR10
            return CIFAR10(root=os.path.join(_WORKSPACE_DIR, 'data'), train=True, download=False, transform=transform)
        except Exception:
            try:
                from torchvision.datasets import CIFAR10
                return CIFAR10(root=os.path.join(_WORKSPACE_DIR, 'data'), train=True, download=True, transform=transform)
            except Exception as e:
                logger.warning(f"Failed to auto-download CIFAR-10 ({e}). Using a synthetic dummy image dataset.")
                return DummyImageDataset(num_samples=1000, img_shape=(3, 32, 32))
    elif isinstance(image_dataset, (str, Path)):
        if str(image_dataset).strip().lower() == 'dummy':
            return DummyImageDataset(num_samples=1000, img_shape=(3, 32, 32))
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


class RenewSemanticCommunicationDataset(Dataset):
    """
    RENEW RF 채널 데이터와 ADJSCC를 결합하여 Semantic Communication (SC) 무선 전송을 모사하는 데이터셋 클래스.

    주요 사양:
    1) 48개 서브캐리어 사용: 원본 52개 서브캐리어(-26 ~ +26, DC 제외) 중 파일럿(-21, -7, 7, 21번, 인덱스 [5, 19, 32, 46])을 제외한 48개 데이터 서브캐리어 추출.
    2) 단일 안테나(SISO) 및 시간 슬라이싱: 로드된 파일의 안테나 중 1개를 선택하고, SC latent 전송에 필요한 시간 스텝 T만큼 자른 t개의 non-overlapping time chunk 중 하나를 선택.
    3) 고정 고유 조합(Unique Pairing): [image, RF channel 파일, 선택 안테나, 시간 청크, natural noise seed]가 seed에 의해 사전 생성되어 학습 내내 고정된 조합으로 유지.
    4) 출력 데이터 shape: [real/imag, T, C] = [2, T, 48] (0: 실수부, 1: 허수부).
    5) SNR 산출: 선택된 RF 채널 슬라이스와 natural noise sigma를 이용하여 계산 후 ADJSCC Encoder에 입력.
    6) SC Encoder는 사전에 학습된 상태로 가정되며, GPU에서 inference mode로만 동작하고 절대 학습(가중치 업데이트)되지 않음.
    7) 시간 스텝 T는 adjscc_config에 설정된 이미지 크기와 압축 수준(tcn 또는 bandwidth ratio)에 의해 자동 계산 지원.
    8) slow_fading_eq 옵션: ADJSCC의 slow_fading_eq처럼 수신 신호 Y에 알려진 채널 H를 나누어 신호 자체(Z)에는 채널 영향을 제거하고, 노이즈가 H로 나누어져 차원(시간, 서브캐리어)에 따라 노이즈 세기(sigma)가 달라지는 신호를 생성하여 diffusion이 학습할 수 있도록 지원.
    """

    def __init__(
        self,
        adjscc_config: Union[str, Path, Dict[str, Any]],
        renew_path: Optional[str] = None,
        path: Optional[str] = None,
        renew_dataset: Optional[renewRfProcessedDataset] = None,
        renew_kwargs: Optional[Dict[str, Any]] = None,
        image_dataset: Optional[Any] = None,
        adjscc_model_path: Optional[str] = None,
        T: Optional[int] = None,
        dataset_len: Optional[int] = None,
        seed: int = 42,
        matching_mode: str = 'interpolate',  # 'interpolate', 'pad_or_slice', 'repeat'
        power_normalize_latent: bool = True,
        view_as_complex: bool = False,       # 기본값 False: [2, T, 48] 실수/허수 채널 반환
        channel_type: Optional[str] = None,  # 'slow_fading' 또는 'slow_fading_eq'
        slow_fading_eq: bool = False,        # slow_fading_eq 활성화 플래그
        eq_epsilon: float = 1e-6,            # 채널 나눗셈 분모 안정화 상수
        # RENEW 관련 옵션
        must_contain: Optional[str] = None,
        must_not_contain: Optional[str] = None,
        sigma: float = 0.1,
        additive_noise_sigma: float = 0.0,
        multiply_noise_sigma: float = 1.0,
        only_additive_noise: bool = False,
        noise_mean_flag: Optional[bool] = None,
        dynamic_noise_sigma: Optional[bool] = None,
        noise_mean_alter_way: bool = False,
        corruption_probability_per_image: float = 1.0,
        corruption_probability_per_pixel: float = 1.0,
        image_corruption_seed: int = 112154,
        image_noise_seed: int = 445481,
        dataset_keep_percentage: float = 1.0,
        flip_keep_dataset: bool = False,
        normalize_value: float = 1.0,
        use_labels: bool = False,
        snr_range: Optional[Union[str, Tuple[float, float], List[float]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **super_kwargs
    ):
        super().__init__()

        if renew_path is None and path is not None:
            renew_path = path
        self.path = renew_path

        self.seed = int(seed)
        self.use_labels = bool(use_labels)
        self.dataset_keep_percentage = float(dataset_keep_percentage)

        self.matching_mode = matching_mode
        self.power_normalize_latent = power_normalize_latent
        self.view_as_complex = view_as_complex
        self.normalize_value = float(normalize_value)
        self.minimum_sigma = float(sigma)
        self.must_contain = must_contain
        self.must_not_contain = must_not_contain
        self.additive_noise_sigma = float(additive_noise_sigma)
        self.multiply_noise_sigma = float(multiply_noise_sigma)
        self.only_additive_noise = bool(only_additive_noise)
        self.noise_mean_alter_way = bool(noise_mean_alter_way)
        self.corruption_probability_per_image = float(corruption_probability_per_image)
        self.corruption_probability_per_pixel = float(corruption_probability_per_pixel)
        self.image_corruption_seed = int(image_corruption_seed)
        self.image_noise_seed = int(image_noise_seed)
        self.flip_keep_dataset = bool(flip_keep_dataset)
        self.snr_range = parse_snr_range(snr_range)

        # 1. ADJSCC 설정 로드 및 채널 모드 결정
        self._config_file_path = Path(adjscc_config) if isinstance(adjscc_config, (str, Path)) else None
        self.adjscc_config = _load_adjscc_config(adjscc_config)
        self.tcn = int(self.adjscc_config.get('transmit_channel_num', 16))
        self.in_channels = int(self.adjscc_config.get('in_channels', 3))
        self.load_model_path = adjscc_model_path or self.adjscc_config.get('load_model_path', None)

        # 채널 유형 결정 ('slow_fading' vs 'slow_fading_eq')
        if slow_fading_eq:
            self.channel_type = 'slow_fading_eq'
        elif channel_type is not None:
            self.channel_type = str(channel_type).lower().strip()
        elif 'channel_type' in self.adjscc_config:
            self.channel_type = str(self.adjscc_config['channel_type']).lower().strip()
        else:
            self.channel_type = 'slow_fading'

        self.eq_epsilon = float(eq_epsilon)

        # dynamic_noise_sigma 및 noise_mean_flag 처리
        # slow_fading_eq는 차원별 다양한 노이즈(varying dimensional noise)를 사용하므로
        # 명시적으로 지정되지 않은 경우 기본적으로 dynamic_noise_sigma=True (noise_mean_flag=False)로 설정
        if dynamic_noise_sigma is not None:
            self.noise_mean_flag = not dynamic_noise_sigma
        elif noise_mean_flag is not None:
            self.noise_mean_flag = bool(noise_mean_flag)
        elif self.channel_type == 'slow_fading_eq':
            self.noise_mean_flag = False
        else:
            self.noise_mean_flag = True
        self.dynamic_noise_sigma = not self.noise_mean_flag

        # 2. RENEW 기본 데이터셋 및 파일 초기화 (기존 RENEW 클래스/함수 직접 활용)
        if renew_dataset is not None:
            self.base_renew = renew_dataset
            self._renew_files = list(renew_dataset._fname)
        elif self.path is not None:
            self.base_renew = renewRfProcessedDataset(
                path=self.path,
                must_contain=self.must_contain,
                must_not_contain=self.must_not_contain,
                sigma=self.minimum_sigma,
                additive_noise_sigma=self.additive_noise_sigma,
                multiply_noise_sigma=self.multiply_noise_sigma,
                only_additive_noise=self.only_additive_noise,
                noise_mean_flag=self.noise_mean_flag,
                noise_mean_alter_way=self.noise_mean_alter_way,
                corruption_probability_per_image=self.corruption_probability_per_image,
                corruption_probability_per_pixel=self.corruption_probability_per_pixel,
                image_corruption_seed=self.image_corruption_seed,
                image_noise_seed=self.image_noise_seed,
                dataset_keep_percentage=self.dataset_keep_percentage,
                flip_keep_dataset=self.flip_keep_dataset,
                normalize_value=self.normalize_value,
                use_labels=self.use_labels,
                snr_range=self.snr_range,
                view_as_complex=True,
                **(renew_kwargs or {})
            )
            self._renew_files = list(self.base_renew._fname)
        else:
            raise ValueError("Either 'renew_dataset' or 'renew_path' (or 'path') must be provided.")

        self._renew_files = sorted(list(self._renew_files))
        if len(self._renew_files) == 0:
            raise ValueError("No RENEW .npz files found matching the criteria!")

        # 기존 RENEW collate functions 직접 재사용
        self.corruption_collate_fn = self.base_renew.corruption_collate_fn
        self.noise_mean_collate_fn = self.base_renew.noise_mean_collate_fn

        # 첫 번째 파일 검사하여 기본 차원(Frame, Antenna, Subcarrier) 확인
        sample_csi, sample_noise = self.base_renew._load_and_normalize(self._renew_files[0])
        self.n_frames, self.n_antennas, self.raw_subcarriers = sample_csi.shape[:3]

        if device is not None:
            self.device = torch.device(device)
        elif 'device' in self.adjscc_config and self.adjscc_config['device'] is not None:
            cfg_dev = str(self.adjscc_config['device']).strip().lower()
            if cfg_dev in ['cuda', 'gpu', 'auto']:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(self.adjscc_config['device'])
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')

        if Attention_Encoder is None:
            raise ImportError("Failed to import Attention_Encoder from misc/ADJSCC/util_module.py.")

        # 동일한 seed로 인스턴스를 생성했을 때 모델 가중치(체크포인트 부재 시)도 동일하게 초기화되도록 시드 고정
        torch.manual_seed(self.seed)
        self.encoder = Attention_Encoder(tcn=self.tcn, in_channels=self.in_channels)

        # ADJSCC 인코더 체크포인트 탐색 및 로드
        resolved_model_path = find_adjscc_checkpoint(
            adjscc_config=self.adjscc_config,
            explicit_path=self.load_model_path,
            config_file_path=self._config_file_path
        )

        if resolved_model_path is not None and resolved_model_path.exists():
            logger.info(f"Loading ADJSCC encoder weights from {resolved_model_path}")
            checkpoint = torch.load(resolved_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'encoder' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                enc_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.')}
                self.encoder.load_state_dict(enc_dict if enc_dict else checkpoint, strict=False)
            else:
                self.encoder.load_state_dict(checkpoint)
            logger.info(f"Successfully loaded ADJSCC encoder weights from {resolved_model_path.name}")
        else:
            logger.info("No ADJSCC model checkpoint specified or found. Initialized Attention_Encoder with fixed seed in eval mode.")

        # SC Encoder는 GPU에 배치되고 eval() 및 requires_grad=False로 완전히 동결됨 (학습되지 않음)
        self.encoder.to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 3. 이미지 데이터셋 초기화
        self.image_dataset = self._init_image_dataset(image_dataset)

        # 4. 시간 스텝 T 결정: T가 지정되지 않은 경우 adjscc_config의 이미지 크기와 압축 수준(tcn 또는 bandwidth ratio)으로 자동 계산
        if T is not None:
            self.T = int(T)
        else:
            sample_img_shape = None
            if len(self.image_dataset) > 0:
                try:
                    sample_item = self.image_dataset[0]
                    if isinstance(sample_item, (tuple, list)):
                        s_img = sample_item[0]
                    elif isinstance(sample_item, dict):
                        s_img = sample_item.get('image', sample_item.get('img'))
                    else:
                        s_img = sample_item
                    if isinstance(s_img, torch.Tensor):
                        sample_img_shape = (s_img.shape[-2], s_img.shape[-1])
                    elif hasattr(s_img, 'shape'):
                        sample_img_shape = (s_img.shape[0], s_img.shape[1])
                    elif hasattr(s_img, 'size'):
                        sample_img_shape = (s_img.size[1], s_img.size[0])
                except Exception:
                    pass
            self.T = calculate_sc_t_from_config(self.adjscc_config, image_shape=sample_img_shape, num_subcarriers=48)

        # 프레임 내 non-overlapping 청크 수 계산
        self.n_time_chunks = max(1, self.n_frames // self.T)

        # 5. [image, RF channel, antenna, natural noise] 고유 조합 (Pairing Table) 생성
        if dataset_len is not None:
            self._dataset_len = int(dataset_len)
        else:
            # 기본적으로 (파일 수 * 시간 청크 수)를 데이터셋 길이로 설정
            self._dataset_len = len(self._renew_files) * self.n_time_chunks

        if self.dataset_keep_percentage < 1.0:
            self._dataset_len = max(1, int(self._dataset_len * self.dataset_keep_percentage))
        self.pairing_table = self._build_pairing_table(self._dataset_len, self.seed)

    @property
    def name(self) -> str:
        return 'RENEW_SC'

    @property
    def image_shape(self) -> List[int]:
        if self.view_as_complex:
            return [self.T, 48]
        return [2, self.T, 48]

    @property
    def num_channels(self) -> int:
        if self.view_as_complex:
            return 1
        return 2

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.T, 48)

    @property
    def label_shape(self) -> List[int]:
        if self.use_labels:
            return self.image_shape
        return [0,]

    @property
    def label_dim(self) -> int:
        if self.use_labels:
            return self.num_channels
        return 0

    @property
    def has_labels(self) -> bool:
        return self.use_labels

    @property
    def calculate_normalized_value(self) -> Tuple[float, float]:
        return (1.0, 1.0)


    def _load_raw_file(self, f_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """npz 파일에서 csi와 noise 배열을 로드하고 정규화합니다 (기존 RENEW 함수 위임)."""
        return self.base_renew._load_and_normalize(f_path)

    def _extract_48_subcarriers(self, csi_data: np.ndarray) -> np.ndarray:
        """
        52개 서브캐리어 중 파일럿 서브캐리어인 -21, -7, 7, 21번(인덱스 [5, 19, 32, 46])을
        제외하여 48개의 데이터 서브캐리어를 추출합니다.
        """
        n_sub = csi_data.shape[-1]
        if n_sub == 48:
            return csi_data
        elif n_sub == 52:
            return csi_data[..., DATA_SUBCARRIER_INDICES_48]
        elif n_sub > 48:
            # 52개 이상의 경우 처음 52개에서 파일럿 제외하거나 앞 48개 슬라이싱
            if n_sub >= 52:
                base = csi_data[..., :52]
                return base[..., DATA_SUBCARRIER_INDICES_48]
            return csi_data[..., :48]
        else:
            raise ValueError(f"Expected at least 48 subcarriers, got {n_sub}")

    def _init_image_dataset(self, image_dataset: Optional[Any]):
        return init_image_dataset(image_dataset)

    def _build_pairing_table(self, length: int, seed: int) -> List[Dict[str, Any]]:
        """
        [image, RF channel 파일, 사용할 antenna, time chunk, natural noise seed]를
        시드 기반으로 고유하고 결정론적으로 조합한 테이블을 생성합니다.
        """
        rng = np.random.RandomState(seed)
        num_files = len(self._renew_files)
        num_images = len(self.image_dataset)

        table = []
        for i in range(length):
            # 파일, 안테나, 시간 청크, 이미지, 노이즈 시드를 결정론적으로 할당
            file_idx = int(rng.randint(0, num_files))
            antenna_idx = int(rng.randint(0, self.n_antennas))
            time_chunk_idx = int(rng.randint(0, self.n_time_chunks))
            image_idx = int(rng.randint(0, num_images))
            noise_seed = int(rng.randint(0, 2**31 - 1))

            table.append({
                'file_idx': file_idx,
                'antenna_idx': antenna_idx,
                'time_chunk_idx': time_chunk_idx,
                'image_idx': image_idx,
                'noise_seed': noise_seed,
            })
        return table

    def __len__(self) -> int:
        return self._dataset_len

    def _match_latent_shape(self, z_complex: torch.Tensor, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        1D 복소수 심볼 텐서를 target_shape (T, 48)의 복소수 numpy 배열로 매칭합니다.
        """
        target_numel = int(np.prod(target_shape))
        dim_z = z_complex.shape[0]

        if dim_z == target_numel:
            z_matched = z_complex.reshape(target_shape).cpu().numpy()
        elif self.matching_mode == 'interpolate':
            real_1d = z_complex.real.unsqueeze(0).unsqueeze(0)
            imag_1d = z_complex.imag.unsqueeze(0).unsqueeze(0)
            real_interp = F.interpolate(real_1d, size=target_numel, mode='linear', align_corners=False).squeeze()
            imag_interp = F.interpolate(imag_1d, size=target_numel, mode='linear', align_corners=False).squeeze()
            z_interp = torch.complex(real_interp, imag_interp)
            z_matched = z_interp.reshape(target_shape).cpu().numpy()
        elif self.matching_mode == 'pad_or_slice':
            if dim_z < target_numel:
                padding = torch.zeros(target_numel - dim_z, dtype=z_complex.dtype, device=z_complex.device)
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

        power_z = np.mean(np.abs(z_matched) ** 2)
        if power_z > 1e-12:
            z_matched = z_matched / np.sqrt(power_z)

        return z_matched

    def _extract_rx_noise(
        self,
        corr_res: Dict[str, Any],
        h_clean: np.ndarray,
        noise_slice: np.ndarray,
        noise_seed: int,
    ) -> np.ndarray:
        """
        RENEW corruption_collate_fn의 결과로부터 무선 채널 전송에 적용할 rx_noise를 직접 추출 및 재사용합니다.
        불필요한 2중 가우시안 난수 생성을 제거하고, RENEW 노이즈 모듈의 결과를 채널 전송 노이즈로 1회만 단일 적용합니다.
        """
        eff_noise_sigma = corr_res['sigma']
        corr_image = corr_res['image']
        corr_noise = corr_image - h_clean

        # only_additive_noise인 경우 corr_noise의 표준편차가 이미 eff_noise_sigma와 정확히 일치
        if self.only_additive_noise:
            return corr_noise.astype(np.complex64)

        # only_additive_noise가 아닌 경우 corruption_collate_fn은 sqrt(eff_sigma^2 - natural_sigma^2) 만큼만 가산하므로
        # 단위 복소 가우시안 노이즈를 복원한 뒤 eff_noise_sigma를 곱하여 전체 채널 노이즈를 산출
        sigma_added_sq = np.maximum(0.0, eff_noise_sigma ** 2 - noise_slice ** 2)
        sigma_added = np.sqrt(sigma_added_sq)

        valid_mask = (sigma_added > 1e-12)
        if np.all(valid_mask):
            unit_noise = corr_noise / sigma_added
            return (unit_noise * eff_noise_sigma).astype(np.complex64)
        elif np.any(valid_mask):
            unit_noise = np.zeros_like(corr_noise)
            unit_noise[valid_mask] = corr_noise[valid_mask] / sigma_added[valid_mask]
            # 미가산 영역(자연 노이즈만 존재하는 영역)은 동일 시드 generator로 보충
            torch_gen = torch.Generator().manual_seed(int(noise_seed + self.image_noise_seed + 4454))
            nr = torch.randn(h_clean.shape, generator=torch_gen).numpy()
            ni = torch.randn(h_clean.shape, generator=torch_gen).numpy()
            backup_unit = (nr + 1j * ni) / np.sqrt(2.0)
            unit_noise[~valid_mask] = backup_unit[~valid_mask]
            return (unit_noise * eff_noise_sigma).astype(np.complex64)
        else:
            # corruption이 가산되지 않은 경우 (corruption_label=0 또는 additive_noise=0)
            if np.max(eff_noise_sigma) < 1e-12:
                return np.zeros_like(h_clean, dtype=np.complex64)
            torch_gen = torch.Generator().manual_seed(int(noise_seed + self.image_noise_seed + 4454))
            nr = torch.randn(h_clean.shape, generator=torch_gen).numpy()
            ni = torch.randn(h_clean.shape, generator=torch_gen).numpy()
            unit_noise = (nr + 1j * ni) / np.sqrt(2.0)
            return (unit_noise * eff_noise_sigma).astype(np.complex64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairing_table[idx]
        file_idx = pair['file_idx']
        antenna_idx = pair['antenna_idx']
        time_chunk_idx = pair['time_chunk_idx']
        image_idx = pair['image_idx']
        noise_seed = pair['noise_seed']

        item_fname = self._renew_files[file_idx]

        # -------------------------------------------------------------
        # 1) RF 채널 로드, 1개 안테나 및 T step 선택, 48개 서브캐리어 추출
        # -------------------------------------------------------------
        raw_csi, raw_noise = self.base_renew._load_and_normalize(item_fname)
        # raw_csi shape: (n_frames, n_antennas, n_subcarriers)
        # raw_noise shape: (n_frames, n_antennas, 1)

        # 시간 청크 범위 슬라이싱 (T 스텝)
        f_start = time_chunk_idx * self.T
        f_end = f_start + self.T

        # 파일 내 프레임 수가 T보다 적거나 슬라이싱 범위가 부족한 경우 처리
        if f_end <= raw_csi.shape[0]:
            h_slice = raw_csi[f_start:f_end, antenna_idx, :]
            n_slice = raw_noise[f_start:f_end, antenna_idx, :]
        else:
            # 부족한 경우 끝부분 슬라이스 후 시간 패딩
            h_sub = raw_csi[:, antenna_idx, :]
            n_sub = raw_noise[:, antenna_idx, :]
            pad_len = self.T - h_sub.shape[0]
            h_slice = np.pad(h_sub, ((0, pad_len), (0, 0)), mode='edge')
            n_slice = np.pad(n_sub, ((0, pad_len), (0, 0)), mode='edge')

        # 52개 중 파일럿 제외하여 48개 서브캐리어 추출 -> shape: (T, 48) complex
        h_clean_48 = self._extract_48_subcarriers(h_slice)
        # n_slice: (T, 1) -> (T, 48)
        noise_slice_48 = np.broadcast_to(n_slice, (self.T, 48)).copy()

        # -------------------------------------------------------------
        # 2) RENEW corruption_collate_fn을 통한 노이즈 계산 및 SNR 산출
        # -------------------------------------------------------------
        corr_item = {
            'image': h_clean_48.copy(),
            'sigma': noise_slice_48.copy(),
            'idx': noise_seed,
        }
        corr_res = self.corruption_collate_fn([corr_item])[0]
        eff_noise_sigma = corr_res['sigma']  # (T, 48)
        corruption_label = corr_res.get('corruption_label', 0)

        # ADJSCC 인코더 입력용 SNR 계산
        sig_pow = float(np.mean(np.abs(h_clean_48) ** 2))
        noise_pow = float(np.mean(eff_noise_sigma ** 2))
        snr_db = float(10.0 * np.log10(sig_pow / max(noise_pow, 1e-12)))

        # -------------------------------------------------------------
        # 3) 고정 매핑된 이미지 선택 및 ADJSCC Encoder를 통한 latent 생성
        # -------------------------------------------------------------
        img_item = self.image_dataset[image_idx]
        if isinstance(img_item, (tuple, list)):
            img, label = img_item[0], img_item[1]
        elif isinstance(img_item, dict):
            img = img_item.get('image', img_item.get('img'))
            label = img_item.get('label', 0)
        else:
            img, label = img_item, 0

        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        if img.ndim == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img_tensor = img.float()

        img_batch = img_tensor.unsqueeze(0).to(self.device)
        snr_batch = torch.tensor([[snr_db]], dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            features = self.encoder(img_batch, snr_batch)

        # -------------------------------------------------------------
        # 4) Latent 복소수 변환, 단위 전력 정규화, (T, 48) reshape
        # -------------------------------------------------------------
        f = features.reshape(1, -1)
        dim_z = f.shape[1] // 2
        z_in = torch.complex(f[:, :dim_z], f[:, dim_z:])

        if self.power_normalize_latent:
            norm_factor = torch.sum(z_in.real ** 2 + z_in.imag ** 2, dim=1, keepdim=True)
            scale = torch.sqrt(dim_z / torch.clamp(norm_factor, min=1e-8))
            z_in = z_in * torch.complex(scale, torch.zeros_like(scale))

        z_complex_1d = z_in.squeeze(0)
        # (T, 48) 복소수 심볼로 매칭
        z_matched = self._match_latent_shape(z_complex_1d, (self.T, 48))

        # -------------------------------------------------------------
        # 5) 무선 채널 전송 및 RENEW NoiseMeanCollateFn 적용
        # -------------------------------------------------------------
        # corruption_collate_fn에서 기생성된 노이즈를 재사용하여 중복 난수 생성 방지
        rx_noise = self._extract_rx_noise(corr_res, h_clean_48, noise_slice_48, noise_seed)

        if self.channel_type == 'slow_fading_eq':
            # Zero-Forcing 채널 등화 (ADJSCC slow_fading_eq 모사):
            # 수신된 신호에 미리 알고 있는 정확한 H를 나누어 신호 자체(Z)에서 채널 영향을 제거:
            # Y_eq = (H * Z + N) / H = Z + N / H
            h_mag = np.abs(h_clean_48)
            eps = self.eq_epsilon
            h_safe = np.where(h_mag < eps, eps * np.exp(1j * np.angle(h_clean_48)), h_clean_48)

            eq_noise = rx_noise / h_safe
            rx_signal = z_matched + eq_noise
            raw_sigma = (eff_noise_sigma / np.maximum(h_mag, eps)).astype(np.float32)
        else:
            # 기본 slow_fading: Y = H * Z + N
            rx_noiseless = h_clean_48 * z_matched  # (T, 48) complex
            rx_signal = rx_noiseless + rx_noise
            raw_sigma = eff_noise_sigma.astype(np.float32)

        # RENEW의 NoiseMeanCollateFn 적용 (noise_mean_flag 및 noise_mean_alter_way 반영)
        mean_res = self.noise_mean_collate_fn([{'sigma': raw_sigma}])[0]['sigma']

        if self.noise_mean_flag:
            # scalar sigma (dynamic_noise_sigma=False)
            out_sigma = np.float32(mean_res)
        else:
            # dynamic_noise_sigma=True: 차원별 노이즈 세기 유지 [2, T, 48] 또는 [T, 48]
            if self.view_as_complex:
                out_sigma = mean_res.astype(np.float32)
            else:
                out_sigma = np.stack([mean_res, mean_res], axis=0).astype(np.float32)

        # -------------------------------------------------------------
        # 6) 출력 형태: [img/real, T, C] = [2, T, 48]
        #    (0: 실수부 Real, 1: 허수부 Imag)
        # -------------------------------------------------------------
        if self.view_as_complex:
            out_image = rx_signal.astype(np.complex64)
            out_clean_channel = h_clean_48.astype(np.complex64)
            out_latent = z_matched.astype(np.complex64)
        else:
            out_image = np.stack([rx_signal.real, rx_signal.imag], axis=0).astype(np.float32)
            out_clean_channel = np.stack([h_clean_48.real, h_clean_48.imag], axis=0).astype(np.float32)
            out_latent = np.stack([z_matched.real, z_matched.imag], axis=0).astype(np.float32)

        if self.use_labels:
            if self.channel_type == 'slow_fading_eq':
                out_label = out_latent
            else:
                out_label = out_clean_channel
        else:
            out_label = np.zeros([0,], dtype=np.float32)

        return {
            'image': out_image,
            'clean_channel': out_clean_channel,
            'latent': out_latent,
            'clean_latent': out_latent,
            'label': out_label,
            'sigma': out_sigma,
            'snr_db': np.float32(snr_db),
            'source_image': img_tensor,
            'source_label': label,
            'antenna_idx': antenna_idx,
            'time_chunk_idx': time_chunk_idx,
            'filename': item_fname,
            'idx': idx,
            'corruption_label': corruption_label,
            'axis_name': ['complex', 'time', 'channel'] if not self.view_as_complex else ['time', 'channel'],
            'additive_noise_sigma': np.float32(self.additive_noise_sigma),
            'channel_type': self.channel_type,
            'dynamic_noise_sigma': self.dynamic_noise_sigma,
        }



# 별칭 지원
renewSemanticCommunicationDataset = RenewSemanticCommunicationDataset


class RenewChannelDataset(Dataset):
    """
    RENEW RF 채널 데이터에서 (T, 48) clean channel slice를 고속으로 추출하는 경량 데이터셋.
    SC 인코더 추론 및 이미지 로딩 없이, 순수 채널 데이터와 자연 노이즈만 CPU 멀티프로세스 DataLoader로 고속 로드합니다.
    """
    def __init__(
        self,
        renew_path: Optional[str] = None,
        path: Optional[str] = None,
        adjscc_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        image_dataset: Optional[Any] = None,
        T: Optional[int] = None,
        dataset_len: Optional[int] = None,
        seed: int = 42,
        must_contain: Optional[str] = None,
        must_not_contain: Optional[str] = None,
        dataset_keep_percentage: float = 1.0,
        flip_keep_dataset: bool = False,
        normalize_value: float = 1.0,
        view_as_complex: bool = False,
        use_labels: bool = False,
        channel_type: Optional[str] = 'slow_fading',
        slow_fading_eq: bool = False,
        **super_kwargs
    ):
        super().__init__()
        if renew_path is None and path is not None:
            renew_path = path
        self.path = renew_path
        self.seed = int(seed)
        self.use_labels = bool(use_labels)
        self.dataset_keep_percentage = float(dataset_keep_percentage)
        self.normalize_value = float(normalize_value)
        self.view_as_complex = bool(view_as_complex)
        self.must_contain = must_contain
        self.must_not_contain = must_not_contain
        self.flip_keep_dataset = bool(flip_keep_dataset)
        self.channel_type = 'slow_fading_eq' if slow_fading_eq else (channel_type or 'slow_fading')

        # 1. Base RENEW dataset helper for file discovery & reading
        self.base_renew = renewRfProcessedDataset(
            path=self.path,
            normalize_value=self.normalize_value,
            must_contain=self.must_contain,
            must_not_contain=self.must_not_contain,
            dataset_keep_percentage=self.dataset_keep_percentage,
            flip_keep_dataset=self.flip_keep_dataset,
        )
        self._renew_files = self.base_renew._fname
        if len(self._renew_files) == 0:
            raise FileNotFoundError(f"No RENEW npz files found at path: {self.path}")

        sample_csi, sample_noise = self.base_renew._load_and_normalize(self._renew_files[0])
        self.n_frames, self.n_antennas, self.raw_subcarriers = sample_csi.shape[:3]

        # 2. Determine T
        if T is not None:
            self.T = int(T)
        else:
            cfg = _load_adjscc_config(adjscc_config) if adjscc_config is not None else {}
            self.T = calculate_sc_t_from_config(cfg, num_subcarriers=48)

        self.n_time_chunks = max(1, self.n_frames // self.T)

        # 3. Determine number of images for pairing table
        self._num_images = 50000
        if image_dataset is not None:
            try:
                img_ds = init_image_dataset(image_dataset)
                self._num_images = len(img_ds)
            except Exception:
                pass

        # 4. Build Pairing Table
        if dataset_len is not None:
            self._dataset_len = int(dataset_len)
        else:
            self._dataset_len = len(self._renew_files) * self.n_time_chunks

        if self.dataset_keep_percentage < 1.0:
            self._dataset_len = max(1, int(self._dataset_len * self.dataset_keep_percentage))

        self.pairing_table = self._build_pairing_table(self._dataset_len, self.seed, self._num_images)

    def _build_pairing_table(self, length: int, seed: int, num_images: int) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        num_files = len(self._renew_files)
        table = []
        for i in range(length):
            file_idx = int(rng.randint(0, num_files))
            antenna_idx = int(rng.randint(0, self.n_antennas))
            time_chunk_idx = int(rng.randint(0, self.n_time_chunks))
            image_idx = int(rng.randint(0, max(1, num_images)))
            noise_seed = int(rng.randint(0, 2**31 - 1))
            table.append({
                'file_idx': file_idx,
                'antenna_idx': antenna_idx,
                'time_chunk_idx': time_chunk_idx,
                'image_idx': image_idx,
                'noise_seed': noise_seed,
            })
        return table

    @property
    def name(self) -> str:
        return 'RENEW_SC'

    @property
    def image_shape(self) -> List[int]:
        if self.view_as_complex:
            return [self.T, 48]
        return [2, self.T, 48]

    @property
    def num_channels(self) -> int:
        return 1 if self.view_as_complex else 2

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.T, 48)

    @property
    def label_shape(self) -> List[int]:
        if self.use_labels:
            return self.image_shape
        return [0,]

    @property
    def label_dim(self) -> int:
        return self.num_channels if self.use_labels else 0

    @property
    def has_labels(self) -> bool:
        return self.use_labels

    def _extract_48_subcarriers(self, csi_data: np.ndarray) -> np.ndarray:
        n_sub = csi_data.shape[-1]
        if n_sub == 48:
            return csi_data
        elif n_sub == 52:
            return csi_data[..., DATA_SUBCARRIER_INDICES_48]
        elif n_sub > 48:
            if n_sub >= 52:
                base = csi_data[..., :52]
                return base[..., DATA_SUBCARRIER_INDICES_48]
            return csi_data[..., :48]
        else:
            raise ValueError(f"Expected at least 48 subcarriers, got {n_sub}")

    def __len__(self) -> int:
        return self._dataset_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairing_table[idx]
        file_idx = pair['file_idx']
        antenna_idx = pair['antenna_idx']
        time_chunk_idx = pair['time_chunk_idx']
        image_idx = pair['image_idx']
        noise_seed = pair['noise_seed']

        item_fname = self._renew_files[file_idx]
        raw_csi, raw_noise = self.base_renew._load_and_normalize(item_fname)

        f_start = time_chunk_idx * self.T
        f_end = f_start + self.T
        if f_end <= raw_csi.shape[0]:
            h_slice = raw_csi[f_start:f_end, antenna_idx, :]
            n_slice = raw_noise[f_start:f_end, antenna_idx, :]
        else:
            h_sub = raw_csi[:, antenna_idx, :]
            n_sub = raw_noise[:, antenna_idx, :]
            pad_len = self.T - h_sub.shape[0]
            h_slice = np.pad(h_sub, ((0, pad_len), (0, 0)), mode='edge')
            n_slice = np.pad(n_sub, ((0, pad_len), (0, 0)), mode='edge')

        h_clean_48 = self._extract_48_subcarriers(h_slice)
        noise_slice_48 = np.broadcast_to(n_slice, (self.T, 48)).copy()

        if self.view_as_complex:
            out_channel = h_clean_48.astype(np.complex64)
        else:
            out_channel = np.stack([h_clean_48.real, h_clean_48.imag], axis=0).astype(np.float32)

        return {
            'image': out_channel,
            'clean_channel': out_channel,
            'raw_noise': noise_slice_48.astype(np.float32),
            'label': np.zeros([0,], dtype=np.float32),
            'sigma': np.float32(0.0),
            'idx': idx,
            'image_idx': image_idx,
            'noise_seed': noise_seed,
            'antenna_idx': antenna_idx,
            'time_chunk_idx': time_chunk_idx,
            'filename': item_fname,
            'axis_name': ['complex', 'time', 'channel'] if not self.view_as_complex else ['time', 'channel'],
        }


class SCBatchSynthesizer(torch.nn.Module):
    """
    GPU 상에서 배치 단위로:
    1) channel_idx 기반 결정론적 Target SNR 및 유효 노이즈 계산
    2) image_idx 배치에 해당하는 사전 캐싱 이미지 추출
    3) SC Encoder(Attention_Encoder) 배치 일괄 추론 및 Latent z 단위 전력 정규화/차원 매칭
    4) 결정론적 복소 가우시안 노이즈 생성 및 무선 채널 전송 (h*z+n 또는 z+n/h)
    5) 최종 디퓨전 학습용 image, label, sigma 생성
    을 수행하는 고속 GPU 파이프라인.
    """
    def __init__(
        self,
        adjscc_config: Union[str, Path, Dict[str, Any]],
        adjscc_model_path: Optional[str] = None,
        image_dataset: Optional[Any] = None,
        device: Union[str, torch.device] = 'cuda',
        T: int = 14,
        channel_type: Optional[str] = 'slow_fading',
        slow_fading_eq: bool = False,
        matching_mode: str = 'interpolate',
        power_normalize_latent: bool = True,
        view_as_complex: bool = False,
        use_labels: bool = False,
        eq_epsilon: float = 1e-6,
        snr_range: Optional[Union[str, Tuple[float, float], List[float]]] = None,
        additive_noise_sigma: float = 0.0,
        multiply_noise_sigma: float = 1.0,
        only_additive_noise: bool = False,
        corruption_probability_per_image: float = 1.0,
        noise_mean_flag: bool = True,
        noise_mean_alter_way: bool = False,
        dynamic_noise_sigma: bool = False,
        image_corruption_seed: int = 112154,
        image_noise_seed: int = 445481,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.device = torch.device(device)
        self.T = int(T)
        self.channel_type = 'slow_fading_eq' if slow_fading_eq else (channel_type or 'slow_fading')
        self.slow_fading_eq = (self.channel_type == 'slow_fading_eq')
        self.matching_mode = matching_mode
        self.power_normalize_latent = power_normalize_latent
        self.view_as_complex = view_as_complex
        self.use_labels = use_labels
        self.eq_epsilon = float(eq_epsilon)
        self.snr_range = parse_snr_range(snr_range)
        self.additive_noise_sigma = float(additive_noise_sigma)
        self.multiply_noise_sigma = float(multiply_noise_sigma)
        self.only_additive_noise = bool(only_additive_noise)
        self.corruption_probability_per_image = float(corruption_probability_per_image)
        self.noise_mean_flag = bool(noise_mean_flag)
        self.noise_mean_alter_way = bool(noise_mean_alter_way)
        self.dynamic_noise_sigma = bool(dynamic_noise_sigma)
        self.image_corruption_seed = int(image_corruption_seed)
        self.image_noise_seed = int(image_noise_seed)
        self.seed = int(seed)

        # 1. ADJSCC Config & Model
        self._config_file_path = Path(adjscc_config) if isinstance(adjscc_config, (str, Path)) else None
        self.adjscc_config = _load_adjscc_config(adjscc_config)
        self.tcn = int(self.adjscc_config.get('transmit_channel_num', 16))
        self.in_channels = int(self.adjscc_config.get('in_channels', 3))
        self.load_model_path = adjscc_model_path or self.adjscc_config.get('load_model_path', None)

        if Attention_Encoder is None:
            raise ImportError("Failed to import Attention_Encoder from misc/ADJSCC/util_module.py.")

        torch.manual_seed(self.seed)
        self.encoder = Attention_Encoder(tcn=self.tcn, in_channels=self.in_channels)

        resolved_model_path = find_adjscc_checkpoint(
            adjscc_config=self.adjscc_config,
            explicit_path=self.load_model_path,
            config_file_path=self._config_file_path
        )
        if resolved_model_path is not None and resolved_model_path.exists():
            checkpoint = torch.load(resolved_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'encoder' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                enc_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.')}
                self.encoder.load_state_dict(enc_dict if enc_dict else checkpoint, strict=False)
            else:
                self.encoder.load_state_dict(checkpoint)

        self.encoder.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2. Image Dataset Pre-loading & Caching
        raw_img_ds = init_image_dataset(image_dataset)
        self._cache_images(raw_img_ds)

    def _cache_images(self, raw_img_ds: Dataset):
        """이미지 데이터셋을 메모리(GPU 텐서)에 일괄 프리로드하여 캐싱합니다."""
        n_images = len(raw_img_ds)
        sample0 = raw_img_ds[0]
        s_img = sample0[0] if isinstance(sample0, (tuple, list)) else (sample0['image'] if isinstance(sample0, dict) else sample0)
        if not isinstance(s_img, torch.Tensor):
            s_img = T.ToTensor()(s_img)
        img_shape = (s_img.shape[-3], s_img.shape[-2], s_img.shape[-1])

        cached = torch.empty((n_images, *img_shape), dtype=torch.float32)
        for i in range(n_images):
            item = raw_img_ds[i]
            img = item[0] if isinstance(item, (tuple, list)) else (item['image'] if isinstance(item, dict) else item)
            if not isinstance(img, torch.Tensor):
                img = T.ToTensor()(img)
            if img.ndim == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            cached[i] = img.float()

        self.register_buffer('cached_images', cached.to(self.device), persistent=False)

    def _compute_deterministic_noise_sigma(
        self,
        h_complex: torch.Tensor,
        noise_sigma_data: torch.Tensor,
        idx_list: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = h_complex.shape[0]

        # Fast CPU random parameters per sample (no tensor touches)
        is_corrupt_list = []
        target_snr_list = []
        for idx in idx_list:
            np_gen = np.random.default_rng(int(self.image_corruption_seed + 175 + int(idx)))
            corrupt = bool(np_gen.random() < self.corruption_probability_per_image)
            is_corrupt_list.append(corrupt)
            if self.snr_range is not None:
                snr_val = float(np_gen.uniform(self.snr_range[0], self.snr_range[1]))
            else:
                snr_val = 0.0
            target_snr_list.append(snr_val)

        is_corrupt = torch.tensor(is_corrupt_list, dtype=torch.bool, device=self.device).view(B, 1, 1)
        target_snr_db = torch.tensor(target_snr_list, dtype=torch.float32, device=self.device).view(B, 1, 1)

        h_complex = h_complex.to(torch.complex64)
        noise_sigma_data = noise_sigma_data.float()

        # Vectorized GPU tensor computation (Zero CPU-GPU stalls, zero .item() calls)
        sig_pow = torch.mean(torch.abs(h_complex) ** 2, dim=(-2, -1), keepdim=True)

        if self.snr_range is not None:
            ratio_snr = 10.0 ** (target_snr_db / 10.0)
            target_sigma_rms = torch.sqrt(sig_pow / torch.clamp(ratio_snr, min=1e-12))

            cur_sigma_rms = torch.sqrt(torch.mean(noise_sigma_data ** 2, dim=(-2, -1), keepdim=True))
            safe_cur_rms = torch.clamp(cur_sigma_rms, min=1e-12)
            scaled_noise = (noise_sigma_data / safe_cur_rms) * target_sigma_rms
            target_noise_sigma = torch.where(cur_sigma_rms > 1e-12, scaled_noise, target_sigma_rms.expand_as(noise_sigma_data))

            if self.only_additive_noise:
                corrupt_sigma = target_noise_sigma
            else:
                corrupt_sigma = torch.maximum(target_noise_sigma, noise_sigma_data)
        else:
            target_noise_sigma = noise_sigma_data * self.multiply_noise_sigma + self.additive_noise_sigma
            if self.only_additive_noise:
                corrupt_sigma = target_noise_sigma - noise_sigma_data
            else:
                corrupt_sigma = target_noise_sigma

        uncorrupt_sigma = torch.zeros_like(noise_sigma_data) if self.only_additive_noise else noise_sigma_data
        eff_noise_sigma = torch.where(is_corrupt, corrupt_sigma, uncorrupt_sigma).float()

        noise_pow = torch.mean(eff_noise_sigma ** 2, dim=(-2, -1), keepdim=True)
        snr_db_tensor = (10.0 * torch.log10(sig_pow / torch.clamp(noise_pow, min=1e-12)).squeeze(-1)).float()

        return eff_noise_sigma, snr_db_tensor

    def _generate_deterministic_noise(
        self,
        shape: Tuple[int, ...],
        noise_seeds: List[int],
        eff_noise_sigma: torch.Tensor
    ) -> torch.Tensor:
        B = shape[0]
        rx_noises = []
        for b in range(B):
            seed_val = int(noise_seeds[b] + self.image_noise_seed + 4454)
            gen = torch.Generator().manual_seed(seed_val)
            nr = torch.randn(shape[1:], generator=gen)
            ni = torch.randn(shape[1:], generator=gen)
            unit_noise = (nr + 1j * ni) / np.sqrt(2.0)
            rx_noises.append(unit_noise)

        rx_unit = torch.stack(rx_noises, dim=0).to(self.device).cfloat()
        rx_noise = rx_unit * eff_noise_sigma
        return rx_noise

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.__call__(batch)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raw_h = batch.get('image', batch.get('clean_channel'))
        if not torch.is_tensor(raw_h):
            raw_h = torch.from_numpy(raw_h)
        raw_h = raw_h.to(self.device)

        raw_noise = batch['raw_noise']
        if not torch.is_tensor(raw_noise):
            raw_noise = torch.from_numpy(raw_noise)
        raw_noise = raw_noise.to(self.device).float()

        idx_list = batch['idx']
        if torch.is_tensor(idx_list):
            idx_list = idx_list.tolist()

        image_idx = batch['image_idx']
        if not torch.is_tensor(image_idx):
            image_idx = torch.tensor(image_idx, dtype=torch.long, device=self.device)
        else:
            image_idx = image_idx.to(device=self.device, dtype=torch.long)
        image_idx = image_idx % self.cached_images.shape[0]

        noise_seeds = batch['noise_seed']
        if torch.is_tensor(noise_seeds):
            noise_seeds = noise_seeds.tolist()

        if raw_h.ndim == 4: # (B, 2, T, 48)
            h_complex = torch.complex(raw_h[:, 0].float(), raw_h[:, 1].float())
        else:
            h_complex = raw_h.to(torch.complex64)

        B, T, C = h_complex.shape

        # 1) Deterministic Effective Noise & SNR
        eff_noise_sigma, snr_batch = self._compute_deterministic_noise_sigma(h_complex, raw_noise, idx_list)
        snr_batch = snr_batch.float()

        # 2) Image Batch Extraction
        img_batch = self.cached_images[image_idx].float()

        # 3) SC Encoder Batch Forward
        with torch.inference_mode():
            features = self.encoder(img_batch, snr_batch)

        # 4) Latent Complex Conversion & Power Normalization & Shape Matching
        f = features.reshape(B, -1)
        dim_z = f.shape[1] // 2
        z_in = torch.complex(f[:, :dim_z], f[:, dim_z:])

        if self.power_normalize_latent:
            norm_factor = torch.sum(z_in.real ** 2 + z_in.imag ** 2, dim=1, keepdim=True)
            scale = torch.sqrt(dim_z / torch.clamp(norm_factor, min=1e-8))
            z_in = z_in * scale

        target_numel = T * C
        if dim_z == target_numel:
            z_matched = z_in.reshape(B, T, C)
        elif self.matching_mode == 'interpolate':
            real_1d = z_in.real.unsqueeze(1)
            imag_1d = z_in.imag.unsqueeze(1)
            real_interp = F.interpolate(real_1d, size=target_numel, mode='linear', align_corners=False).squeeze(1)
            imag_interp = F.interpolate(imag_1d, size=target_numel, mode='linear', align_corners=False).squeeze(1)
            z_matched = torch.complex(real_interp, imag_interp).reshape(B, T, C)
        elif self.matching_mode == 'pad_or_slice':
            if dim_z < target_numel:
                padding = torch.zeros((B, target_numel - dim_z), dtype=z_in.dtype, device=z_in.device)
                z_res = torch.cat([z_in, padding], dim=1)
            else:
                z_res = z_in[:, :target_numel]
            z_matched = z_res.reshape(B, T, C)
        elif self.matching_mode == 'repeat':
            repeat_factor = int(np.ceil(target_numel / dim_z))
            z_rep = z_in.repeat(1, repeat_factor)[:, :target_numel]
            z_matched = z_rep.reshape(B, T, C)
        else:
            raise ValueError(f"Unknown matching_mode: {self.matching_mode}")

        power_z = torch.mean(torch.abs(z_matched) ** 2, dim=(-2, -1), keepdim=True)
        z_matched = z_matched / torch.sqrt(torch.clamp(power_z, min=1e-12))

        # 5) Deterministic Gaussian Noise N & Channel Reception
        rx_noise = self._generate_deterministic_noise((B, T, C), noise_seeds, eff_noise_sigma)

        if self.channel_type == 'slow_fading_eq':
            h_mag = torch.abs(h_complex)
            eps = self.eq_epsilon
            h_safe = torch.where(h_mag < eps, eps * torch.exp(1j * torch.angle(h_complex)), h_complex)
            eq_noise = rx_noise / h_safe
            rx_signal = z_matched + eq_noise
            raw_sigma = (eff_noise_sigma / torch.clamp(h_mag, min=eps)).float()
            clean_target = z_matched
        else:
            rx_noiseless = h_complex * z_matched
            rx_signal = rx_noiseless + rx_noise
            raw_sigma = eff_noise_sigma.float()
            clean_target = h_complex

        # 6) Noise reduction
        if self.noise_mean_flag:
            if self.noise_mean_alter_way:
                out_sigma = torch.sqrt(torch.mean(raw_sigma ** 2, dim=(-2, -1)))
            else:
                out_sigma = torch.mean(raw_sigma, dim=(-2, -1))
        else:
            if self.view_as_complex:
                out_sigma = raw_sigma
            else:
                out_sigma = torch.stack([raw_sigma, raw_sigma], dim=1)

        # 7) Output format
        if self.view_as_complex:
            out_image = rx_signal
            out_clean_channel = h_complex
            out_latent = z_matched
            out_label = clean_target if self.use_labels else torch.zeros((B, 0), device=self.device)
        else:
            out_image = torch.stack([rx_signal.real, rx_signal.imag], dim=1).float()
            out_clean_channel = torch.stack([h_complex.real, h_complex.imag], dim=1).float()
            out_latent = torch.stack([z_matched.real, z_matched.imag], dim=1).float()
            out_label = torch.stack([clean_target.real, clean_target.imag], dim=1).float() if self.use_labels else torch.zeros((B, 0), device=self.device)

        return {
            'image': out_image,
            'label': out_label,
            'sigma': out_sigma,
            'clean_channel': out_clean_channel,
            'latent': out_latent,
            'snr_db': snr_batch.squeeze(1),
        }
