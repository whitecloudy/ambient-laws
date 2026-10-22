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
        """이미지 데이터셋을 로드하거나 초기화합니다 (기본 CIFAR-10)."""
        if image_dataset is None:
            transform = T.Compose([T.ToTensor()])
            # 1. ADJSCC 표준 경로(data/cifar10/train)에 이미 존재하는 CIFAR-10 확인
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

            # 2. torchvision CIFAR10 로컬 캐시(cifar-10-batches-py) 확인
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
