# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import fnmatch

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------


import ambient_utils

from glob import glob
import warnings

class renewRfProcessedDataset(ambient_utils.dataset_utils.Dataset):
    def __init__(self, 
                 path,                   # Path to files.
                 resolution      = None, # Ensure specific resolution, None = highest available.
                 must_contain    = None, # Require filenames to contain this substring.
                 must_not_contain = None, # Require filenames to NOT contain this substring.
                 sigma: float = 0.1,     # ensured minimum currption sigma
                 additive_noise_sigma: float = 0.0,
                 multiply_noise_sigma: float = 1.0,
                 view_as_complex = False,
                 complex_merge_axis = 0,
                 transpose = None,
                 noise_mean_flag = True,
                 noise_mean_alter_way = False,
                 corruption_probability_per_image = 0.0,
                 corruption_probability_per_pixel = 1.0,
                 image_corruption_seed = 112154,
                 image_noise_seed = 445481,
                 dataset_keep_percentage = 1.0,
                 normalize_value = 1.0,
                 use_labels  = False,
                 flip_keep_dataset = False,
                 only_additive_noise = False,
                 **super_kwargs):
        self.minimum_sigma = sigma
        assert additive_noise_sigma >= 0.0, "additive_noise_sigma must be non-negative"
        self.additive_noise_sigma = additive_noise_sigma
        assert multiply_noise_sigma >= 1.0, "multiply_noise_sigma must be greater than or equal to 1.0"
        self.multiply_noise_sigma = multiply_noise_sigma
        self._noise_mean_flag = noise_mean_flag
        self._noise_mean_alter_way = noise_mean_alter_way   # Temporary flag for altering noise mean in a different way, will be removed in the future after testing
        self._path = path
        self._normalize_value = normalize_value
        self._view_as_complex = view_as_complex
        self._complex_merge_axis = complex_merge_axis
        self._label_spliting_flag = use_labels
        self._transpose = tuple(transpose) if transpose is not None else None
        self._all_fnames = []
        self._image_corruption_seed = image_corruption_seed
        self._image_noise_seed = image_noise_seed
        self.corruption_probability_per_image = corruption_probability_per_image
        self.corruption_probability_per_pixel = corruption_probability_per_pixel
        self.only_additive_noise = only_additive_noise

        self._fname = []
        self._axis_name = ['frame', 'antenna', 'channel']   # Default axis names for indexing
        if isinstance(self._path, list):
            for path in self._path:
                self._fname += glob(os.path.join(path, '*.npz'))
        elif os.path.isdir(self._path):
            self._fname = glob(os.path.join(self._path, '*.npz'), recursive=True)
        else:
            raise IOError('Path must point to a directory or list of file paths')
        
        print("Dataset Length before", len(self._fname))
        if must_contain is not None:
            if any(c in must_contain for c in '*?['):
                self._fname = {fname for fname in self._fname if fnmatch.fnmatch(fname, must_contain)}
            else:
                self._fname = {fname for fname in self._fname if must_contain in fname}
        
        if must_not_contain is not None:
            if any(c in must_not_contain for c in '*?['):
                self._fname = {fname for fname in self._fname if not fnmatch.fnmatch(fname, must_not_contain)}
            else:
                self._fname = {fname for fname in self._fname if must_not_contain not in fname}
        print("Dataset Length after", len(self._fname))
        self._fname = list(self._fname)

        if dataset_keep_percentage < 1.0:
            num_to_keep = int(len(self._fname) * dataset_keep_percentage)
            rng = np.random.RandomState(self._image_corruption_seed)
            rng.shuffle(self._fname)
            tmp_fname = self._fname
            self._fname = tmp_fname[:num_to_keep]
            self._removed_fname = tmp_fname[num_to_keep:]
            print(f"Dataset reduced to {len(self._fname)} samples using keep percentage {dataset_keep_percentage}")
        else:
            self._removed_fname = []
            print(f"Dataset not reduced using keep percentage {dataset_keep_percentage}")
        
        self._fname = sorted(self._fname)
        self._removed_fname = sorted(self._removed_fname)

        if flip_keep_dataset:
            tmp_fname = self._fname.copy()
            self._fname = self._removed_fname
            self._removed_fname = tmp_fname

        name = os.path.splitext(os.path.basename(self._path))[0]
        single_csi_shape = self.__load_single_file(self._fname[0])[0].shape

        if self._label_spliting_flag:
            tmp_shape = np.array(single_csi_shape)
            tmp_shape[-1] = tmp_shape[-1] // 2
            single_csi_shape = tuple(tmp_shape.tolist())

        if self._transpose is not None:
            actual_csi_shape = [single_csi_shape[ax] for ax in self._transpose] + [single_csi_shape[2]]
            self._axis_name = [self._axis_name[ax] for ax in self._transpose] + [self._axis_name[2]]
        else:
            actual_csi_shape = list(single_csi_shape)

        if not self._view_as_complex:
            if self._complex_merge_axis is not None:
                actual_csi_shape[self._complex_merge_axis] *= 2
                self._axis_name[self._complex_merge_axis] = 'complex/' + self._axis_name[self._complex_merge_axis]
            else:
                actual_csi_shape.append(2)
                self._axis_name.append('complex')
        elif self._complex_merge_axis is not None:
            warnings.warn("complex_merge_axis is only applicable when view_as_complex is False")

        self.image_actual_shape = actual_csi_shape
        raw_shape = [len(self._fname)] + actual_csi_shape

        self._resolution = raw_shape[-2:]
        super().__init__(name=name, raw_shape=raw_shape, use_labels=use_labels, **super_kwargs)
        if self._label_spliting_flag:
            self._label_shape = actual_csi_shape
        else:
            self._label_shape = [0,]
            
        self.transpose_collate_fn = TransposeCollateFn(self._transpose)
        self.complex_view_collate_fn = ComplexViewCollateFn(self._view_as_complex, self._complex_merge_axis)
        self.corruption_collate_fn = CorruptionCollateFn(
            additive_noise_sigma=self.additive_noise_sigma,
            multiply_noise_sigma=self.multiply_noise_sigma,
            corruption_probability_per_image=self.corruption_probability_per_image,
            only_additive_noise=self.only_additive_noise,
            image_corruption_seed=self._image_corruption_seed,
            image_noise_seed=self._image_noise_seed
        )
        self.noise_mean_collate_fn = NoiseMeanCollateFn(self._noise_mean_flag, self._noise_mean_alter_way)

    def __load_single_file(self, f_path):
        """
        Loads CSI and noise sigma data from a .npz file.

        Args:
            f_path (str): Path to the .npz file containing the data.

        Returns:
            tuple:
                - csi_data (numpy.ndarray): The CSI (Channel State Information) data array.
                - noise_sigma_data (numpy.ndarray): The noise sigma data array.

        Note:
            The .npz file is expected to contain two arrays:
                - 'csi': numpy.ndarray, shape and dtype depend on dataset.
                - 'noise': numpy.ndarray, shape and dtype depend on dataset.
        """
        npz_data = np.load(f_path)
        csi_data = np.array(npz_data['csi'])
        noise_sigma_data = np.array(npz_data['noise'])

        return csi_data, noise_sigma_data

    def __len__(self):
        return len(self._fname)
    
    def _load_and_normalize(self, item_fname):
        csi_data, noise_sigma_data = self.__load_single_file(item_fname)
        # csi_data : (frame, antenna, channel) - complex
        # noise_sigma_data : (frame, antenna) - float

        # Scale Normalization
        csi_data /= self._normalize_value
        noise_sigma_data /= self._normalize_value

        noise_sigma_data = np.expand_dims(noise_sigma_data, -1)
        # noise_sigma_data : (frame, antenna, 1) - float
        return csi_data, noise_sigma_data

    def _apply_transpose(self, csi_data, noise_sigma_data):
        if self._transpose is not None:
            assert noise_sigma_data.ndim == len(self._transpose)+1, f"noise_sigma_data ndim and transpose length mismatch, {noise_sigma_data.ndim} != {len(self._transpose)+1}"
            item = {'image': csi_data, 'label': np.array([]), 'sigma': noise_sigma_data}
            item = self.transpose_collate_fn([item])[0]
            csi_data = item['image']
            noise_sigma_data = item['sigma']
        return csi_data, noise_sigma_data

    def _handle_complex_view(self, csi_data, noise_sigma_data):
        item = {'image': csi_data, 'label': np.array([]), 'sigma': noise_sigma_data}
        item = self.complex_view_collate_fn([item])[0]
        csi_data = item['image']
        noise_sigma_data = item['sigma']
        return csi_data, noise_sigma_data

    def _apply_corruption(self, csi_data, noise_sigma_data, idx):
        item = {'image': csi_data, 'sigma': noise_sigma_data, 'idx': idx}
        item = self.corruption_collate_fn([item])[0]
        return item['image'], item['sigma'], item['corruption_label']

    def _split_label(self, csi_data):
        if self._label_spliting_flag:
            csi_data, label_data = np.split(csi_data, 2, axis=2)
        else:
            label_data = np.zeros([self.image_shape[0], 0], dtype=np.float32)
        return csi_data, label_data

    def _format_output(self, csi_data, label_data, noise_sigma_data, item_fname, corruption_label):
        if self._view_as_complex:
            dtype = np.complex64
        else:
            dtype = np.float32

        return {
            'image': csi_data.astype(dtype),
            "label": label_data.astype(dtype),
            'sigma': noise_sigma_data.astype(np.float32),
            'filename': item_fname,
            "noise": np.random.randn(*csi_data.shape),
            'corruption_label': corruption_label,
            'additive_noise_sigma': self.additive_noise_sigma,
            'complex_merge_axis': self._complex_merge_axis,
            'axis_name': self._axis_name,
        }

    def __create_item__(self, item_fname, idx):
        csi_data, noise_sigma_data = self._load_and_normalize(item_fname)
        csi_data, noise_sigma_data = self._apply_transpose(csi_data, noise_sigma_data)
        csi_data, noise_sigma_data = self._handle_complex_view(csi_data, noise_sigma_data)
        csi_data, noise_sigma_data, corruption_label = self._apply_corruption(csi_data, noise_sigma_data, idx)
        
        if self._noise_mean_flag:
            item = {'sigma': noise_sigma_data}
            item = self.noise_mean_collate_fn([item])[0]
            noise_sigma_data = item['sigma']

        csi_data, label_data = self._split_label(csi_data)

        return self._format_output(csi_data, label_data, noise_sigma_data, item_fname, corruption_label)

    def __getitem__(self, idx):
        item_fname = self._fname[idx]

        return_item = self.__create_item__(item_fname, idx)
        return_item['idx'] = idx
        return return_item

    
    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[1:]

    @property
    def label_shape(self):
        if self._label_spliting_flag:
            return list(self._label_shape)
        else:
            return [0,]

    @property
    def label_dim(self):
        if self._label_spliting_flag:
            return self.label_shape[0]
        else:
            return 0

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64
    
    @property
    def get_normalize_value(self):
        return self._normalize_value
    
    @property
    def calculate_normalized_value(self):
        var_sum = 0.0
        var_count = 0

        for fname in self._fname:
            csi_data, noise_sigma_data = self._load_and_normalize(fname)
            var_sum += (np.sum((csi_data * np.conj(csi_data)).real.flatten()))
            var_count += csi_data.flatten().size
        return np.sqrt(var_sum / var_count)
    
class TransposeCollateFn(object):
    """
    데이터 배치의 'image', 'label', 'sigma' 텐서에 대해 축 교환(transpose)을 적용하는 Collate Function.
    """
    def __init__(self, transpose=None):
        self._transpose = tuple(transpose) if transpose is not None else None
        self.__name__ = "TransposeCollateFn"

    def __call__(self, batch):
        if self._transpose is None:
            return batch
        
        for item in batch:
            # image와 label은 설정된 transpose 축을 적용하고 나머지 축들은 그대로 유지
            img_ndim = item['image'].ndim
            transpose_axes = self._transpose + tuple(range(len(self._transpose), img_ndim))
            item['image'] = np.transpose(item['image'], transpose_axes)
            
            if item['label'].size > 0:
                lbl_ndim = item['label'].ndim
                lbl_axes = self._transpose + tuple(range(len(self._transpose), lbl_ndim))
                item['label'] = np.transpose(item['label'], lbl_axes)
            
            if item['sigma'].ndim >= len(self._transpose):
                sig_ndim = item['sigma'].ndim
                sig_axes = self._transpose + tuple(range(len(self._transpose), sig_ndim))
                item['sigma'] = np.transpose(item['sigma'], sig_axes)
            
            if 'noise' in item:
                noise_ndim = item['noise'].ndim
                noise_axes = self._transpose + tuple(range(len(self._transpose), noise_ndim))
                item['noise'] = np.transpose(item['noise'], noise_axes)

            if 'axis_name' in item:
                old_axis_name = list(item['axis_name'])
                new_axis_name = [old_axis_name[ax] for ax in self._transpose]
                for i in range(len(old_axis_name)):
                    if i not in self._transpose:
                        new_axis_name.append(old_axis_name[i])
                item['axis_name'] = new_axis_name
        return batch


class ComplexViewCollateFn(object):
    """
    복소수 형태의 데이터를 실수 형태로 변환하는 Collate Function.
    view_as_complex가 False일 경우, 실수부와 허수부를 분리하여 지정된 축에 합칩니다.
    """
    def __init__(self, view_as_complex=False, complex_merge_axis=0):
        self._view_as_complex = view_as_complex
        self._complex_merge_axis = complex_merge_axis
        self.__name__ = "ComplexViewCollateFn"

    def __call__(self, batch):
        if self._view_as_complex:
            for item in batch:
                for key in ['image', 'label']:
                    if item[key].size > 0:
                        item[key] = item[key].astype(np.complex64)
                item['sigma'] = item['sigma'].astype(np.float32)
            return batch
        
        for item in batch:
            # image와 label 데이터를 처리
            for key in ['image', 'label']:
                if item[key].size == 0 or not np.iscomplexobj(item[key]):
                    item[key] = item[key].astype(np.float32)
                    continue
                
                data = item[key]
                if self._complex_merge_axis is not None:
                    data = np.append(data.real, data.imag, axis=self._complex_merge_axis)
                else:
                    data = np.expand_dims(data, -1)
                    data = np.append(data.real, data.imag, axis=-1)
                item[key] = data.astype(np.float32)

            # sigma 데이터를 처리 (복소수가 아니므로 값을 복제하여 차원을 맞춤)
            sigma = item['sigma']
            if self._complex_merge_axis is not None:
                sigma = np.append(sigma, sigma, axis=self._complex_merge_axis)
            else:
                sigma = np.expand_dims(sigma, -1)
                sigma = np.append(sigma, sigma, axis=-1)
            item['sigma'] = sigma.astype(np.float32)

            # 'noise'가 있다면 새로운 shape에 맞게 다시 생성
            if 'noise' in item:
                item['noise'] = np.random.randn(*item['image'].shape)

            if 'axis_name' in item:
                axis_name = list(item['axis_name'])
                if self._complex_merge_axis is not None:
                    axis_name[self._complex_merge_axis] = 'complex/' + axis_name[self._complex_merge_axis]
                else:
                    axis_name.append('complex')
                item['axis_name'] = axis_name

        return batch


class CorruptionCollateFn(object):
    """
    이미지에 노이즈를 추가하여 손상시키는 Collate Function.
    additive_noise_sigma와 multiply_noise_sigma를 이용해 노이즈 레벨을 조절합니다.
    """
    def __init__(self, additive_noise_sigma=0.0, multiply_noise_sigma=1.0, 
                 corruption_probability_per_image=0.0, only_additive_noise=False,
                 image_corruption_seed=112154, image_noise_seed=445481):
        self.additive_noise_sigma = additive_noise_sigma
        self.multiply_noise_sigma = multiply_noise_sigma
        self.corruption_probability_per_image = corruption_probability_per_image
        self.only_additive_noise = only_additive_noise
        self._image_corruption_seed = image_corruption_seed
        self._image_noise_seed = image_noise_seed
        self.__name__ = "CorruptionCollateFn"

    def __call__(self, batch):
        if not ((self.additive_noise_sigma > 0.0 or self.multiply_noise_sigma > 1.0) and self.corruption_probability_per_image > 0.0):
            for item in batch:
                if self.only_additive_noise:
                    item['sigma'] = np.zeros_like(item['sigma'])
                item['corruption_label'] = 0
            return batch

        for item in batch:
            idx = item['idx']
            csi_data = item['image']
            noise_sigma_data = item['sigma']

            np_gen = np.random.default_rng(int(self._image_corruption_seed + 175 + idx))
            
            if np_gen.random() < self.corruption_probability_per_image:
                torch_gen = torch.Generator()
                torch_gen.manual_seed(int(idx + self._image_noise_seed + 4454))

                corruption_label = 1
                noise_image = torch.randn(csi_data.shape, generator=torch_gen).numpy()
                
                target_noise_sigma = noise_sigma_data * self.multiply_noise_sigma + self.additive_noise_sigma

                if self.only_additive_noise:
                    target_noise_sigma = target_noise_sigma - noise_sigma_data
                    sigma_will_be_added = target_noise_sigma
                    final_noise_sigma = target_noise_sigma
                else:
                    # 추가될 노이즈의 분산 = 목표 분산 - 현재 분산
                    sigma_will_be_added_squared = (target_noise_sigma ** 2) - (noise_sigma_data ** 2)
                    # 분산이 음수가 되는 경우를 방지 (수치적 오류 등)
                    sigma_will_be_added = np.sqrt(np.maximum(0, sigma_will_be_added_squared))
                    final_noise_sigma = target_noise_sigma

                csi_data = csi_data + (noise_image * sigma_will_be_added).astype(csi_data.dtype)
                noise_sigma_data = final_noise_sigma
            else:
                if self.only_additive_noise:
                    noise_sigma_data = np.zeros_like(noise_sigma_data)
                corruption_label = 0

            item['image'] = csi_data
            item['sigma'] = noise_sigma_data
            item['corruption_label'] = corruption_label

        return batch


class NoiseMeanCollateFn(object):
    """
    noise_sigma_data를 평균 내어 단일 스칼라 값으로 만드는 Collate Function.
    """
    def __init__(self, noise_mean_flag=True, noise_mean_alter_way=False):
        self.noise_mean_flag = noise_mean_flag
        self.noise_mean_alter_way = noise_mean_alter_way
        self.__name__ = "NoiseMeanCollateFn"

    def __call__(self, batch):
        if not self.noise_mean_flag:
            return batch
        
        for item in batch:
            if self.noise_mean_alter_way:
                item['sigma'] = np.array(np.sqrt(np.mean(np.power(item['sigma'], 2))), dtype=np.float32)
            else:
                item['sigma'] = np.array(np.mean(item['sigma']), dtype=np.float32)
        return batch



class rf_augmentation_collate_fn(object):
    def __init__(self, flip_probability=0.0, phase_shift_probability=0.0, ant_axis=None, complex_axis=None, other_collate_fn=[]):
        self._flip_probability = flip_probability
        self._phase_shift_probability = phase_shift_probability
        self._ant_axis = ant_axis
        self._complex_axis = complex_axis   # complex_axis is only valid when batch datas are not complex.
        self._other_collate_fn = other_collate_fn

        self._name = str(["rf_augmentation_collate_fn",] + [collate_fn.__name__ for collate_fn in self._other_collate_fn])

    def __call__(self, batch):
        if self._phase_shift_probability > 0.0:
            batch = self.random_phase_shift_collate_fn(batch)
        if self._flip_probability > 0.0:
            batch = self.antenna_random_flip_collate_fn(batch)
        
        for collate_fn in self._other_collate_fn:
            batch = collate_fn(batch)
        return batch
    
    @property
    def __name__(self):
        return self._name
    
    def random_phase_shift_collate_fn(self, batch):
        """
        배치 내의 각 샘플에 대해 랜덤한 위상 이동을 적용하는 collate_fn입니다.
        """
        shifted_batch = []

        does_complex = np.iscomplexobj(batch[0]['image'])
        axis_name_list = batch[0]['axis_name']

        complex_axis = None
        if not does_complex:
            if self._complex_axis is not None:
                complex_axis = self._complex_axis
            else:
                for idx, axis_name in enumerate(axis_name_list):
                    if 'complex' in axis_name:
                        complex_axis = idx # 개별 이미지 단위로 처리하므로 +1을 제외합니다.
                        break
                assert complex_axis is not None, "No antenna axis found in axis_name"
                self._complex_axis = complex_axis
        else:
            complex_axis = None


        for item in batch:
            img = item['image']
            noise_sigma = item['sigma']
            label = item['label']

            if complex_axis is not None:
                # 복소수 형태로 변환
                real_part, imag_part = np.split(img, 2, axis=complex_axis)
                img = real_part + 1j * imag_part

                real_part, imag_part = np.split(label, 2, axis=complex_axis)
                label = real_part + 1j * imag_part

            # 랜덤한 위상 이동 생성 (0에서 2π 사이)
            random_phase = np.random.uniform(0, 2 * np.pi)

            # 이미지에 위상 이동 적용 (복소수 데이터라고 가정)
            shifted_img = img * np.exp(1j * random_phase)
            shifted_label = label * np.exp(1j * random_phase)

            if complex_axis is not None:
                # 다시 실수 형태로 변환
                shifted_img = np.stack((shifted_img.real, shifted_img.imag), axis=complex_axis)
                shifted_label = np.stack((shifted_label.real, shifted_label.imag), axis=complex_axis)

            item['image'] = shifted_img
            item['label'] = shifted_label

            shifted_batch.append(item)

        return shifted_batch

    def antenna_random_flip_collate_fn(self, batch):
        """
        배치 내의 각 샘플에 대해 안테나 축을 랜덤하게 뒤집는 collate_fn입니다.
        """
        if self._ant_axis is not None:
            ant_axis = self._ant_axis
        else:   # ant_axis가 명시적으로 지정되지 않은 경우, axis_name을 통해 안테나 축을 찾습니다.
            axis_name_list = batch[0]['axis_name']
            ant_axis = None
            for idx, axis_name in enumerate(axis_name_list):
                if 'antenna' in axis_name:
                    ant_axis = idx # 개별 이미지 단위로 처리하므로 +1을 제외합니다.
                    break
            assert ant_axis is not None, "No antenna axis found in axis_name"
            self._ant_axis = ant_axis

        ant_axis_size = batch[0]['image'].shape[ant_axis]

        # 안테나 축에 복소수 정보가 포함된 경우, 실수와 허수 부분을 동일하게 뒤집기 위해 complex_axis를 설정합니다.
        if self._complex_axis is None: 
            axis_name_list = batch[0]['axis_name']
            if 'complex' in axis_name_list[ant_axis]:
                ant_axis_complex = True
                ant_axis_size = ant_axis_size//2
                self._complex_axis = ant_axis
            else:
                ant_axis_complex = False
        else:
            if self._complex_axis == ant_axis:
                ant_axis_complex = True
                ant_axis_size = ant_axis_size//2
            else:
                ant_axis_complex = False

        flipped_batch = []

        for item in batch:
            if (self._flip_probability > 0.0) and (np.random.rand() < self._flip_probability):
                img = item['image']
                noise_sigma = item['sigma']
                label = item['label']
            
                flip_idx = np.random.permutation(ant_axis_size)

                # 안테나 축에 복소수 정보가 포함된 경우, 실수와 허수 부분을 동일하게 뒤집기 위해 인덱스를 확장합니다.
                if ant_axis_complex:
                    flip_idx = np.concatenate((flip_idx, flip_idx+ant_axis_size))

                # np.take를 사용하여 ant_axis를 기준으로 데이터 순서를 변경합니다.
                item['image'] = np.take(img, flip_idx, axis=ant_axis)
                if item['sigma'].ndim > ant_axis:
                    if item['sigma'].shape[ant_axis] == item['image'].shape[ant_axis]:
                        item['sigma'] = np.take(noise_sigma, flip_idx, axis=ant_axis)
                    else:
                        warnings.warn(f"Sigma shape {noise_sigma.shape} does not match expected antenna axis size {ant_axis_size}, skipping sigma flip")
                if label.shape[0] != 0:
                    item['label'] = np.take(label, flip_idx, axis=ant_axis)
                item['flip_aug'] = 1

                flipped_batch.append(item)
            else:
                item['flip_aug'] = 0
                flipped_batch.append(item)

        return flipped_batch

import pandas as pd

import torch.nn.functional as F
def _power2ceil(original):
    return int(2**np.ceil(np.log2(original)))

def pad_collate_fn(batch):
    """
    가변 크기의 'image' 텐서를 패딩하여 동일한 크기로 맞춘 후,
    하나의 배치로 합칩니다.
    """
    # 1. 배치 내에서 'image'의 최대 높이와 너비를 찾습니다.
    # 각 image는 (C, H, W) 형태라고 가정합니다.
    max_h = max(item['image'].shape[-2] for item in batch)
    max_w = max(item['image'].shape[-1] for item in batch)

    # 2. 높이와 너비를 2의 배수로 올림합니다.
    target_h = _power2ceil(max_h)
    target_w = _power2ceil(max_w)

    # 3. 각 'image'를 목표 크기로 패딩합니다.
    padded_batch = []

    for item in batch:
        img = item['image']
        label = item['label']
        original_shape = torch.tensor(img.shape)
        # (padding_left, padding_right, padding_top, padding_bottom)
        pad_h = target_h - img.shape[1]
        pad_w = target_w - img.shape[2]
        # img의 차원 수에 따라 pad_width를 동적으로 생성
        pad_width = [(0, 0)] * (img.ndim - 2) + [(0, pad_h), (0, pad_w)]
        padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)

        item['original_shape'] = original_shape
        item['image'] = padded_img

        if img.shape == label.shape:
            padded_label = np.pad(label, pad_width, mode='constant', constant_values=0)
            item['label'] = padded_label

        padded_batch.append(item)

    return padded_batch

def widar_collate_fn(batch):
    """
    가변 크기의 'image' 텐서를 패딩하여 동일한 크기로 맞춘 후,
    하나의 배치로 합칩니다.
    """
    padded_batch = []
    target_h = 2048

    for item in batch:
        img = item['image']
        noise = item['noise']
        if img.shape[-2] > target_h:
            img = img[:, :target_h, :]
            noise = noise[:, :target_h, :]
            original_shape = torch.tensor(img.shape)    # save original shape after cutting
        elif img.shape[-2] < target_h:
            pad_h = target_h - img.shape[-2]
            pad_width = [(0, 0)] * (img.ndim - 2) + [(0, pad_h), (0, 0)]
            original_shape = torch.tensor(img.shape)    # save original shape before padding
            img = np.pad(img, pad_width, mode='constant', constant_values=0)
            noise = np.pad(noise, pad_width, mode='constant', constant_values=0)
        else:
            original_shape = torch.tensor(img.shape)

        item['original_shape'] = original_shape
        item['image'] = img
        item['noise'] = noise
        padded_batch.append(item)

    # 4. 다른 데이터들도 배치로 만듭니다.
    collated_batch = torch.utils.data.default_collate(padded_batch)

    # # 5. 패딩된 이미지들을 쌓아서(stack) 배치에 추가합니다.
    # collated_batch['image'] = torch.stack(images)
    # collated_batch['original_shape'] = torch.stack(original_shapes)

    return collated_batch



class widarRfDataset(ambient_utils.dataset_utils.Dataset):
    def __init__(self, 
                 path,                   # Path to files.
                 resolution      = None, # Ensure specific resolution, None = highest available.
                 must_contain    = None, # Require filenames to contain this substring.
                 must_not_contain = None, # Require filenames to NOT contain this substring.
                 sigma: float = 0.1,     # ensured minimum currption sigma
                 utilize_remaining_frame = False,
                 view_as_complex = False,
                 complex_merge_axis = 0,
                 transpose = None,
                 noise_mean_flag = True,
                 corruption_probability_per_image = 0.0,
                 corruption_probability_per_pixel = 1.0,
                 image_corruption_seed = 112154,
                 image_noise_seed = 445481,
                 normalize_value = 1.0,
                 label_key = ["gesture", ],
                 **super_kwargs):
        self.minimum_sigma = sigma
        self._noise_mean_flag = noise_mean_flag
        self._path = path
        self._normalize_value = normalize_value

        if not os.path.isdir(self._path):
            raise IOError('Path must point to a directory or list of file paths')

        # key => [date, user, gesture, torso, face, repetition, rx, file name, room]
        # date : data captured date
        # gesture : gesture while data capture
        # torso : torso location
        # face : where user faced
        # rx : rx device number
        self.data_label_df = self._load_label_datafile(self._path)
        self._add_room_number()
        
        # label로 사용할 key값 선택
        self.cond_label = label_key
        self.max_value =  self._load_max_value(self.data_label_df, self.cond_label)
        self.min_value =  self._load_min_value(self.data_label_df, self.cond_label)
        
        # 사용할 데이터 선택
        if must_contain is not None:
            self.data_label_df = self.data_label_df[self.data_label_df['file name'].str.contains(must_contain)]
        
        if must_not_contain is not None:
            self.data_label_df = self.data_label_df[~self.data_label_df['file name'].str.contains(must_not_contain)]

        self._prefix_fname = sorted(list(self.data_label_df['file name'].unique()))

        if type(resolution) is tuple or type(resolution) is list:
            self._frame_resolution = resolution[0]
            self._ant_resolution = resolution[1]
            self._channel_resolution = resolution[2] if len(resolution) > 2 else None
        elif type(resolution) is int:
            self._frame_resolution = resolution
            self._ant_resolution = resolution
            self._channel_resolution = None
        else:
            assert False, "resolution must be int or tuple/list of int"
            
        # # data shape = (frame, user, antenna, channel)
        # self._csi_raw_data_list = [np.load(fprefix+'.csi.npy', mmap_mode='r') for fprefix in self._prefix_fname]
        # self._noise_raw_data_list = [np.load(fprefix+'.noise.npy', mmap_mode='r') for fprefix in self._prefix_fname]
        # # print(self._csi_raw_data_list[0].shape)
        
        self._view_as_complex = view_as_complex
        self._complex_merge_axis = complex_merge_axis
        self._transpose = tuple(transpose) if transpose is not None else None

        name = os.path.splitext(os.path.basename(self._path))[0]
        single_csi_shape = [self._frame_resolution, self._ant_resolution, self._channel_resolution]
        if self._transpose is not None:
            actual_csi_shape = [single_csi_shape[ax] for ax in self._transpose] + [single_csi_shape[2]]
        else:
            actual_csi_shape = single_csi_shape

        if not self._view_as_complex:
            if self._complex_merge_axis is not None:
                actual_csi_shape[self._complex_merge_axis] *= 2
        elif self._complex_merge_axis is not None:
            import warnings
            warnings.warn("complex_merge_axis is only applicable when view_as_complex is False")

        self.image_actual_shape = actual_csi_shape
        raw_shape = [len(self.data_label_df)] + actual_csi_shape

        self._resolution = raw_shape[-2:]
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _condition_maker(self, label : pd.Series):
        cond_list = []
        for key in self.cond_label:
            value = label[key] - self.min_value[key]
            value_range = self.max_value[key] - self.min_value[key] + 1
            cond_frac = np.zeros((value_range))
            cond_frac[value] = 1

            cond_list.append(cond_frac)

        return np.concatenate(cond_list)



    def _load_max_value(self, df, cond_label):
        max_list = {}
        for label in cond_label:
            max_list[label] = max(df[label])

        return pd.Series(max_list)
    
    def _load_min_value(self, df, cond_label):
        min_list = {}
        for label in cond_label:
            min_list[label] = min(df[label])

        return pd.Series(min_list)


    def _add_room_number(self):
        data_room_matching = {
            20181109: 1,
            20181112: 1,
            20181115: 1,
            20181116: 1,
            20181117: 2,
            20181118: 2,
            20181121: 1,
            20181127: 2,
            20181128: 2,
            20181130: 1,
            20181204: 2,
            20181205: 2,
            20181208: 2,
            20181209: 2,
            20181211: 3
        }
        self.data_label_df["room"] = [data_room_matching[date] for date in self.data_label_df["date"]]

    def _load_label_datafile(self, save_dir) -> pd.DataFrame:
        data_label_df_list = pd.read_pickle(save_dir+"/labels.pkl")
        return data_label_df_list
    
    @staticmethod
    def csi_data_loader(data_dir : str) -> np.ndarray:
        return np.load(data_dir+".npz", allow_pickle=False)

    def __len__(self):
        return len(self.data_label_df)

    def __getitem__(self, idx):
        # idx_tuple = self._each_data_idx[idx]
        # csi_data = self._csi_raw_data_list[idx_tuple[0]][idx_tuple[1]: idx_tuple[1]+self._frame_resolution,
        #                                                  idx_tuple[2],
        #                                                  idx_tuple[3]: idx_tuple[3]+self._ant_resolution] / self._normalize_value
        # noise_sigma_data = self._noise_raw_data_list[idx_tuple[0]][idx_tuple[1]: idx_tuple[1]+self._frame_resolution,
        #                                                      idx_tuple[2],
        #                                                      idx_tuple[3]: idx_tuple[3]+self._ant_resolution] / self._normalize_value       
        # # csi_data : (frame, antenna, channel) - complex
        # # noise_sigma_data : (frame, antenna) - float
        idx_fname = self.data_label_df.iloc[idx]["file name"]
        idx_dir = "/".join([self._path, idx_fname])
        with self.csi_data_loader(idx_dir) as data:
            csi_data = data["csi"]
            noise_sigma_data = np.expand_dims(data["noise"], -1)
            time_data = data["time"]
        # csi_data : (frame, antenna, channel) - complex
        # noise_sigma_data : (frame, 1) - float

        if self._transpose is not None:
            assert noise_sigma_data.ndim == len(self._transpose), f"noise_sigma_data ndim and transpose length mismatch {noise_sigma_data.shape} {len(self._transpose)}"
            csi_data = np.transpose(csi_data, self._transpose + (2,))
            noise_sigma_data = np.transpose(noise_sigma_data, self._transpose)

        if not self._view_as_complex:
            csi_data = np.expand_dims(np.array(csi_data), -1)
            csi_data = csi_data.view(np.float32)

            if self._complex_merge_axis is not None:
                csi_data = np.concatenate((np.take(csi_data, 0, axis=-1),
                                           np.take(csi_data, 1, axis=-1)), axis=self._complex_merge_axis)
        
        if self._noise_mean_flag:
            noise_sigma_data = np.mean((noise_sigma_data))

        return {
            'image': csi_data.astype(np.float32),
            "label": np.zeros([self.image_shape[0], 0], dtype=np.float32),
            'sigma': noise_sigma_data.astype(np.float32),
            'idx': idx,
            'filename': idx_fname,
            "noise": np.random.randn(*csi_data.shape),
        }
    
    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[1:]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64
    
    @property
    def get_normalize_value(self):
        return self._normalize_value
    
    @property
    def calculate_normalized_value(self):
        var_sum = 0.0
        var_count = 0

        for csi_data in self._csi_raw_data_list:
            var_sum += (np.sum(np.abs(csi_data)**2))
            var_count += csi_data.size

        return np.sqrt(var_sum / var_count)



if __name__ == "__main__":
    pass