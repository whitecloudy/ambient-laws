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

def find_npz_files(dir_path : str, recursive=True):
    file_paths = []
    if recursive:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.npz'):
                    file_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(dir_path):
            if file.endswith('.npz'):
                file_paths.append(os.path.join(dir_path, file))
    return file_paths

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
                self._fname += find_npz_files(path)
        elif os.path.isdir(self._path):
            self._fname = find_npz_files(self._path)
        else:
            raise IOError('Path must point to a directory or list of file paths')
        
        print("Dataset Length before", len(self._fname))
        if must_contain is not None:
            import re
            if must_contain.startswith('regex:'):
                pattern = re.compile(must_contain.replace('regex:', ''))
                self._fname = {fname for fname in self._fname if pattern.search(fname)}
            elif any(c in must_contain for c in '*?['):
                self._fname = {fname for fname in self._fname if fnmatch.fnmatch(fname, must_contain)}
            else:
                self._fname = {fname for fname in self._fname if must_contain in fname}
        
        if must_not_contain is not None:
            import re
            if must_not_contain.startswith('regex:'):
                pattern = re.compile(must_not_contain.replace('regex:', ''))
                self._fname = {fname for fname in self._fname if not pattern.search(fname)}
            elif any(c in must_not_contain for c in '*?['):
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
        self.curruption_collate_fn = None
        if self._noise_mean_alter_way:
            self.corruption_collate_fn = AlternativeCorruptionCollateFn(
                additive_noise_sigma=self.additive_noise_sigma,
                multiply_noise_sigma=self.multiply_noise_sigma,
                corruption_probability_per_image=self.corruption_probability_per_image,
                only_additive_noise=self.only_additive_noise,
                image_corruption_seed=self._image_corruption_seed,
                image_noise_seed=self._image_noise_seed,
            )
        else:
            self.corruption_collate_fn = CorruptionCollateFn(
                additive_noise_sigma=self.additive_noise_sigma,
                multiply_noise_sigma=self.multiply_noise_sigma,
                corruption_probability_per_image=self.corruption_probability_per_image,
                only_additive_noise=self.only_additive_noise,
                image_corruption_seed=self._image_corruption_seed,
                image_noise_seed=self._image_noise_seed,
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

        if self._complex_merge_axis is None:
            return_complex_merge_axis = -1
        else:
            return_complex_merge_axis = self._complex_merge_axis

        return {
            'image': csi_data.astype(dtype),
            "label": label_data.astype(dtype),
            'sigma': noise_sigma_data.astype(np.float32),
            'filename': item_fname,
            "noise": np.random.randn(*csi_data.shape),
            'corruption_label': corruption_label,
            'additive_noise_sigma': self.additive_noise_sigma,
            'complex_merge_axis': return_complex_merge_axis,
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

        noise_var_sum = 0.0
        noise_var_count = 0

        for fname in tqdm(self._fname, desc="Calculating normalized value"):
            csi_data, noise_sigma_data = self._load_and_normalize(fname)
            var_sum += (np.sqrt((np.sum(csi_data.real.flatten()**2) + np.sum(csi_data.imag.flatten()**2)) / (csi_data.flatten().size * 2)))
            var_count += 1

            noise_var_sum += (np.sqrt(np.sum((noise_sigma_data**2).flatten())/noise_sigma_data.flatten().size))
            noise_var_count += 1

        return (var_sum / var_count), (noise_var_sum / noise_var_count)

    
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
    
def sigma_mean(sigma):
    return np.sqrt(np.mean(np.power(sigma, 2)))
    
class AlternativeCorruptionCollateFn(object):
    """
    이미지에 노이즈를 추가하여 손상시키는 Collate Function.
    additive_noise_sigma와 multiply_noise_sigma를 이용해 노이즈 레벨을 조절합니다.
    """
    def __init__(self, additive_noise_sigma=0.0, multiply_noise_sigma=1.0, 
                 corruption_probability_per_image=0.0, only_additive_noise=False,
                 image_corruption_seed=112154, image_noise_seed=445481):
        self.additive_noise_sigma = additive_noise_sigma
        self.corruption_probability_per_image = corruption_probability_per_image
        self.only_additive_noise = only_additive_noise
        self._image_corruption_seed = image_corruption_seed
        self._image_noise_seed = image_noise_seed
        self.__name__ = "AlternativeCorruptionCollateFn"

    def __call__(self, batch):
        if not ((self.additive_noise_sigma > 0.0) and self.corruption_probability_per_image > 0.0):
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

                sqrt_E_sigma_n_power2 = sigma_mean(noise_sigma_data)

                multiply_noise_sigma = self.additive_noise_sigma/sqrt_E_sigma_n_power2 + 1.0

                target_noise_sigma = noise_sigma_data * multiply_noise_sigma 

                if self.only_additive_noise:
                    # sigma_target = sigma/(E[sigma^2]^0.5) * sigma_additive
                    # sigma/(E[sigma^2]^0.5) : Normalized sigma
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
    def __init__(self, flip_probability=0.0, phase_shift_probability=0.0, ant_axis=None, complex_axis=None, other_collate_fn=[], is_label_complex=False):
        self._flip_probability = flip_probability
        self._phase_shift_probability = phase_shift_probability
        self._ant_axis = ant_axis
        self._complex_axis = complex_axis   # complex_axis is only valid when batch datas are not complex.
        self._is_label_complex = is_label_complex
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

                if self._is_label_complex and label.size > 0 and len(label.shape) > complex_axis:
                    real_part, imag_part = np.split(label, 2, axis=complex_axis)
                    label = real_part + 1j * imag_part

            # 랜덤한 위상 이동 생성 (0에서 2π 사이)
            random_phase = np.random.uniform(0, 2 * np.pi)

            # 이미지에 위상 이동 적용 (복소수 데이터라고 가정)
            shifted_img = img * np.exp(1j * random_phase)
            if label.size > 0:
                shifted_label = label * np.exp(1j * random_phase)
            else:
                shifted_label = label

            if complex_axis is not None:
                # 다시 실수 형태로 변환
                shifted_img = np.concatenate((shifted_img.real, shifted_img.imag), axis=complex_axis)
                if self._is_label_complex and shifted_label.size > 0 and len(shifted_label.shape) > complex_axis:
                    shifted_label = np.concatenate((shifted_label.real, shifted_label.imag), axis=complex_axis)

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
                if label.shape[0] != 0 and len(label.shape) > ant_axis:
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

class pad_collate_fn(object):
    def __init__(self, dynamic_noise=False):
        self.dynamic_noise = dynamic_noise
        self.__name__ = "pad_collate_fn"

    def __call__(self, batch):
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
            pad_h = target_h - img.shape[-2]
            pad_w = target_w - img.shape[-1]
            # img의 차원 수에 따라 pad_width를 동적으로 생성
            pad_width = [(0, 0)] * (img.ndim - 2) + [(0, pad_h), (0, pad_w)]
            padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    
            item['original_shape'] = original_shape
            item['image'] = padded_img
    
            if img.shape == label.shape:
                padded_label = np.pad(label, pad_width, mode='constant', constant_values=0)
                item['label'] = padded_label
                
            if self.dynamic_noise and 'sigma' in item:
                sigma = item['sigma']
                pad_width = []
                for idx, sigma_dim in enumerate(sigma.shape):
                    if sigma_dim == padded_img.shape[idx] or sigma_dim == 1:
                        pad_width.append((0, 0))
                    else:
                        pad_width.append((0, padded_img.shape[idx] - sigma_dim))
                padded_sigma = np.pad(sigma, pad_width, mode='constant', constant_values=0)
                item['sigma'] = padded_sigma
    
            padded_batch.append(item)
    
        return padded_batch

import multiprocessing as mp
from tqdm import tqdm

def _process_widar_file(file_path, ant_size, label_dim):
    with np.load(file_path) as file_data:
        csi_data = file_data['csi_data'].astype(np.complex64)
        csi_data = np.split(csi_data, axis=1, indices_or_sections=ant_size)
        csi_data = np.stack(csi_data)

        noise_sigma_data = file_data['noise_array'].astype(np.float32) 
        # noise_sigma_data = np.repeat(np.expand_dims(noise_sigma_data, axis=0), ant_size, axis=0)
        noise_sigma_data = np.split(noise_sigma_data, axis=1, indices_or_sections=ant_size)
        noise_sigma_data = np.stack(noise_sigma_data)

        true_length = file_data['resized_non_padded_length'].astype(np.float32)

    base_name = os.path.basename(file_path)
    parts = base_name.split('-')
    gesture_num = None
    for part in parts:
        if part.startswith('gesture'):
            gesture_num = int(part.replace('gesture', ''))
            break

    if gesture_num is None:
        label = None
    else:
        label_list_map = [0, 1, 2, 3, 17, 18]
        label = np.zeros(label_dim, dtype=np.float32)
        if gesture_num in label_list_map:
            label[label_list_map.index(gesture_num)] = 1.0

    return csi_data, noise_sigma_data, true_length, label

class WiDARDataset(Dataset):
    def __init__(self, path, 
                 transform=None, 
                 dataset_keep_percentage=0.8, 
                 split_seed=42, 
                 must_contain="regex:gesture(0|1|2|3|17|18)(?!\d)", 
                 must_not_contain=None,
                 normalize_value=1.0,
                 sigma_norm = False,
                 transpose=None,
                 view_as_complex=False,
                 complex_merge_axis=0,
                 additive_noise_sigma: float = 0.0,
                 multiply_noise_sigma: float = 1.0,
                 corruption_probability_per_image = 0.0,
                 corruption_probability_per_pixel = 1.0,
                 only_additive_noise = False,
                 image_corruption_seed = 112154,
                 image_noise_seed = 445481,
                 noise_mean_flag = True,
                 noise_mean_alter_way = True,
                 use_labels=True, 
                 cache = None,
                 sigma = 0.0,
                 only_positive = False,
                 resolution = (3, 256, 30),
                 max_size = None,
                 ):
        self.dir_path = path
        self.transform = transform
        self.split_ratio = dataset_keep_percentage
        self.split_seed = split_seed
        self.must_contain = must_contain
        self.must_not_contain = must_not_contain
        self.normalize_value = normalize_value
        self.sigma_norm = sigma_norm
        self.transpose = tuple(transpose) if transpose is not None else None
        self.view_as_complex = view_as_complex
        self.complex_merge_axis = complex_merge_axis
        self.additive_noise_sigma = additive_noise_sigma
        self.multiply_noise_sigma = multiply_noise_sigma
        self.corruption_probability_per_image = corruption_probability_per_image
        self.only_additive_noise = only_additive_noise
        self.image_corruption_seed = image_corruption_seed
        self.image_noise_seed = image_noise_seed
        self.noise_mean_flag = noise_mean_flag
        self.noise_mean_alter_way = noise_mean_alter_way
        self._use_labels = use_labels
        self.file_paths = []

        if isinstance(self.dir_path, str):
            # find .npz files in recursive way
            self.file_paths = find_npz_files(self.dir_path)
        elif isinstance(self.dir_path, list):
            for path in self.dir_path:
                self.file_paths.extend(find_npz_files(path))
        else:
            raise ValueError("dir_path should be a string or a list of strings.")
        
        if must_contain is not None:
            import re
            if must_contain.startswith('regex:'):
                pattern = re.compile(must_contain.replace('regex:', ''))
                self.file_paths = [fname for fname in self.file_paths if pattern.search(fname)]
            elif any(c in must_contain for c in '*?['):
                self.file_paths = [fname for fname in self.file_paths if fnmatch.fnmatch(fname, must_contain)]
            else:
                self.file_paths = [fname for fname in self.file_paths if must_contain in fname]

        if must_not_contain is not None:
            import re
            if must_not_contain.startswith('regex:'):
                pattern = re.compile(must_not_contain.replace('regex:', ''))
                self.file_paths = [fname for fname in self.file_paths if not pattern.search(fname)]
            elif any(c in must_not_contain for c in '*?['):
                self.file_paths = [fname for fname in self.file_paths if not fnmatch.fnmatch(fname, must_not_contain)]
            else:
                self.file_paths = [fname for fname in self.file_paths if must_not_contain not in fname]

        # self.file_paths = self.filter_files(self.file_paths)

        self._axis_name = ['antenna', 'frame', 'channel']
        self._label_dim = 6
        self._name = "WiDARDataset"
        self._image_shape = resolution 
        actual_csi_shape = self._image_shape

        self.idx_list = np.arange(len(self.file_paths))
        self.live_idx, self.dead_idx = self.split_datasets(self.idx_list)

        # Load one sample to determine the shape of the csi data.
        csi_data, _, _, _ = _process_widar_file(self.file_paths[self.live_idx[0]], 3, self._label_dim)

        self._image_shape = csi_data.shape
        self._file_image_shape = self._image_shape

        if self.transpose is not None:
            actual_csi_shape = self._image_shape
            img_ndim = len(actual_csi_shape)
            # Assumes transpose is a permutation of the first N axes that are being transposed
            transpose_axes = self.transpose + tuple(range(len(self.transpose), img_ndim))
            
            actual_csi_shape = [actual_csi_shape[i] for i in transpose_axes]
            self._axis_name = [self._axis_name[i] for i in transpose_axes]

        if not self.view_as_complex:
            if self.complex_merge_axis is not None:
                actual_csi_shape[self.complex_merge_axis] *= 2
                self._axis_name[self.complex_merge_axis] = 'complex/' + self._axis_name[self.complex_merge_axis]
            else:
                actual_csi_shape.append(2)
                self._axis_name.append('complex')
        
        self._image_shape = tuple(actual_csi_shape)
        self._label_dim = 6
        self._name = "WiDARDataset"
        # --- End of shape calculation logic ---


        self.transpose_collate_fn = TransposeCollateFn(self.transpose)
        self.complex_view_collate_fn = ComplexViewCollateFn(self.view_as_complex, self.complex_merge_axis)
        if self.noise_mean_alter_way:
            self.corruption_collate_fn = AlternativeCorruptionCollateFn(
                additive_noise_sigma=self.additive_noise_sigma,
                multiply_noise_sigma=self.multiply_noise_sigma,
                corruption_probability_per_image=self.corruption_probability_per_image,
                only_additive_noise=self.only_additive_noise,
                image_corruption_seed=self.image_corruption_seed,
                image_noise_seed=self.image_noise_seed,
            )
        else:
            self.corruption_collate_fn = CorruptionCollateFn(
                additive_noise_sigma=self.additive_noise_sigma,
                multiply_noise_sigma=self.multiply_noise_sigma,
                corruption_probability_per_image=self.corruption_probability_per_image,
                only_additive_noise=self.only_additive_noise,
                image_corruption_seed=self.image_corruption_seed,
                image_noise_seed=self.image_noise_seed,
            )
        self.noise_mean_collate_fn = NoiseMeanCollateFn(self.noise_mean_flag, self.noise_mean_alter_way)

    def split_antenna_axis(self, csi_data, antenna_num=3):
        # input csi_data shape : (frame, antenna*channel)
        # output csi_data shape : (frame, antenna, channel)

        csi_data = np.split(csi_data, axis=1, indices_or_sections=antenna_num)
        return np.stack(csi_data)


    def _normalize_data(self, csi_data, noise_sigma_data):

        # if self.sigma_norm:
        #     # Sigma Normalization
        #     csi_data /= np.maximum(noise_sigma_data, 1e-12)
        #     noise_sigma_data = np.ones_like(noise_sigma_data)
        # else:
        #     mean_sigma = np.sqrt(np.mean(noise_sigma_data**2))
        #     csi_data /= mean_sigma
        #     noise_sigma_data /= mean_sigma

        csi_data /= self.normalize_value
        noise_sigma_data /= self.normalize_value

        # fit noise_sigma_data shape to csi_data shape for later processing
        while len(csi_data.shape) > len(noise_sigma_data.shape):
             noise_sigma_data = np.expand_dims(noise_sigma_data, axis=-1)
        # noise_sigma_data : (frame, antenna, 1) - float

        return csi_data, noise_sigma_data

    def _apply_transpose(self, csi_data, noise_sigma_data, axis_name):
        if self.transpose is not None:
            # For WiDAR, label is 1D vector, so we don't transpose it.
            item = {'image': csi_data, 'label': np.array([]), 'sigma': noise_sigma_data, 'axis_name': axis_name}
            item = self.transpose_collate_fn([item])[0]
            csi_data = item['image']
            noise_sigma_data = item['sigma']
            axis_name = item['axis_name']
        return csi_data, noise_sigma_data, axis_name

    def _handle_complex_view(self, csi_data, noise_sigma_data, axis_name):
        # For WiDAR, label is 1D vector, so we don't process it as complex.
        item = {'image': csi_data, 'label': np.array([]), 'sigma': noise_sigma_data, 'axis_name': axis_name}
        item = self.complex_view_collate_fn([item])[0]
        csi_data = item['image']
        noise_sigma_data = item['sigma']
        axis_name = item['axis_name']
        return csi_data, noise_sigma_data, axis_name

    def _apply_corruption(self, csi_data, noise_sigma_data, idx):
        item = {'image': csi_data, 'sigma': noise_sigma_data, 'idx': idx}
        item = self.corruption_collate_fn([item])[0]
        return item['image'], item['sigma'], item['corruption_label']

    def flip_splits(self):
        tmp = self.live_idx
        self.live_idx = self.dead_idx
        self.dead_idx = tmp
    

    def filter_files(self, file_paths):
        filtered_paths = []
        
        must_have = [self.must_have] if isinstance(self.must_have, str) else self.must_have
        must_not_have = [self.must_not_have] if isinstance(self.must_not_have, str) else self.must_not_have
        for path in file_paths:
            os.path.normpath(path)
            if must_have and not any(fnmatch.fnmatch(path, f'*{feature}*') for feature in must_have):
                continue
            if must_not_have and any(fnmatch.fnmatch(path, f'*{feature}*') for feature in must_not_have):
                continue
            filtered_paths.append(path)
        return filtered_paths
    
    def split_datasets(self, file_paths):
        split_shuffler = np.random.default_rng(seed=self.split_seed)
        split_shuffler.shuffle(file_paths)
        split_index = int(self.split_ratio * len(file_paths))
        return file_paths[:split_index], file_paths[split_index:]
    
    def get_gesture_num_from_path(self, file_path):
        # Extract gesture label from file path (assuming format includes '-gestureX-')
        base_name = os.path.basename(file_path)
        parts = base_name.split('-')
        for part in parts:
            if part.startswith('gesture'):
                return int(part.replace('gesture', ''))
        return None  # Return None if no gesture label is found
        
    def __len__(self):
        return len(self.live_idx)
    
    def __getitem__(self, idx):
        true_idx = self.live_idx[idx]
        file_path = self.file_paths[true_idx]

        csi_data, noise_sigma_data, true_length, label_data = _process_widar_file(file_path, 3, self._label_dim)
        assert csi_data.shape == self._file_image_shape, f"Error: Expected CSI data shape {self._file_image_shape}, but got {csi_data.shape} for file {file_path}"
        axis_name = self._axis_name.copy()

        # 1. Normalize
        csi_data, noise_sigma_data = self._normalize_data(csi_data, noise_sigma_data)
        
        # 2. Transpose
        csi_data, noise_sigma_data, axis_name = self._apply_transpose(csi_data, noise_sigma_data, axis_name)

        # 3. Handle complex view
        csi_data, noise_sigma_data, axis_name = self._handle_complex_view(csi_data, noise_sigma_data, axis_name)

        # 4. Apply corruption
        csi_data, noise_sigma_data, corruption_label = self._apply_corruption(csi_data, noise_sigma_data, idx)

        # 5. Noise mean
        if self.noise_mean_flag:
            item = {'sigma': noise_sigma_data}
            item = self.noise_mean_collate_fn([item])[0]
            noise_sigma_data = item['sigma']

        if self.view_as_complex:
            dtype = np.complex64
        else:
            dtype = np.float32
        
        if self.complex_merge_axis is None:
            return_complex_merge_axis = -1
        else:
            return_complex_merge_axis = self.complex_merge_axis

        return {
            'image': csi_data.astype(dtype),
            'sigma': noise_sigma_data.astype(np.float32),
            'label': label_data.astype(np.float32),
            'true_length': true_length,
            'filename': file_path,
            'idx': idx,
            'corruption_label': corruption_label,
            'additive_noise_sigma': self.additive_noise_sigma,
            'complex_merge_axis': return_complex_merge_axis,
            'axis_name': axis_name,
        }
    
    @property
    def __name__(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._image_shape)

    @property
    def resolution(self):
        # Returns the shape of the spatial dimensions (all dimensions except the first one).
        return self.image_shape[1:]

    @property
    def name(self):
        return self._name

    @property
    def label_dim(self):
        return self._label_dim

    @property
    def has_labels(self):
        return True
    
    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]
    
    @property
    def calculate_normalized_value(self):
        var_sum = 0.0
        var_count = 0

        noise_var_sum = 0.0
        noise_var_count = 0

        for idx in tqdm(self.live_idx, desc="Calculating normalized value"):
            file_path = self.file_paths[idx]
            csi_data, noise_sigma_data, _, _ = _process_widar_file(file_path, 3, self._label_dim)
            csi_data, noise_sigma_data = self._normalize_data(csi_data, noise_sigma_data)
            var_sum += (np.sqrt((np.sum(csi_data.real.flatten()**2) + np.sum(csi_data.imag.flatten()**2)) / (csi_data.flatten().size * 2)))
            var_count += 1

            noise_var_sum += (np.sqrt(np.sum((noise_sigma_data**2).flatten())/noise_sigma_data.flatten().size))
            noise_var_count += 1

        return (var_sum / var_count), (noise_var_sum / noise_var_count)



def _process_xrf55_file(file_path, ant_size, label_dim):
    with np.load(file_path) as file_data:
        csi_data = file_data['csi_data'].astype(np.complex64)
        csi_data = np.split(csi_data, axis=1, indices_or_sections=ant_size)
        csi_data = np.stack(csi_data)

        noise_sigma_data = file_data['noise_array'].astype(np.float32) 
        noise_sigma_data = np.split(noise_sigma_data, axis=1, indices_or_sections=ant_size)
        noise_sigma_data = np.stack(noise_sigma_data)

    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    gesture_num = int(parts[1])-1 # gesture number starts with 1 in XRF55 dataset, so we subtract 1 to make it zero-indexed.

    if gesture_num is None:
        label = None
    else:
        label = np.zeros(label_dim, dtype=np.float32)
        label[gesture_num] = 1.0

    return csi_data, noise_sigma_data, None, label


class XRF55Dataset(Dataset):
    def __init__(self, path, 
                 transform=None, 
                 dataset_keep_percentage=0.8, 
                 split_seed=42, 
                 must_contain=None, 
                 must_not_contain=None,
                 normalize_value=1.0,
                 sigma_norm = False,
                 transpose=None,
                 view_as_complex=False,
                 complex_merge_axis=0,
                 additive_noise_sigma: float = 0.0,
                 multiply_noise_sigma: float = 1.0,
                 corruption_probability_per_image = 0.0,
                 corruption_probability_per_pixel = 1.0,
                 only_additive_noise = False,
                 image_corruption_seed = 112154,
                 image_noise_seed = 445481,
                 noise_mean_flag = True,
                 noise_mean_alter_way = True,
                 use_labels=True, # XRF55 seems to always have labels
                 cache = None,
                 sigma = 0.0,
                 only_positive = False,
                 resolution = (3, 500, 90),
                 max_size = None,
                 ):
        self.dir_path = path
        self.transform = transform
        self.split_ratio = dataset_keep_percentage
        self.split_seed = split_seed
        self.must_contain = must_contain
        self.must_not_contain = must_not_contain
        self.normalize_value = normalize_value
        self.sigma_norm = sigma_norm
        self.transpose = tuple(transpose) if transpose is not None else None
        self.view_as_complex = view_as_complex
        self.complex_merge_axis = complex_merge_axis
        self.additive_noise_sigma = additive_noise_sigma
        self.multiply_noise_sigma = multiply_noise_sigma
        self.corruption_probability_per_image = corruption_probability_per_image
        self.only_additive_noise = only_additive_noise
        self.image_corruption_seed = image_corruption_seed
        self.image_noise_seed = image_noise_seed
        self.noise_mean_flag = noise_mean_flag
        self.noise_mean_alter_way = noise_mean_alter_way
        self.use_labels = use_labels
        self.file_paths = []

        if isinstance(self.dir_path, str):
            # find .npz files in recursive way
            self.file_paths = find_npz_files(self.dir_path)
        elif isinstance(self.dir_path, list):
            for path in self.dir_path:
                self.file_paths.extend(find_npz_files(path))
        else:
            raise ValueError("dir_path should be a string or a list of strings.")
        
        if must_contain is not None:
            import re
            if must_contain.startswith('regex:'):
                pattern = re.compile(must_contain.replace('regex:', ''))
                self.file_paths = [fname for fname in self.file_paths if pattern.search(fname)]
            elif any(c in must_contain for c in '*?['):
                self.file_paths = [fname for fname in self.file_paths if fnmatch.fnmatch(fname, must_contain)]
            else:
                self.file_paths = [fname for fname in self.file_paths if must_contain in fname]

        if must_not_contain is not None:
            import re
            if must_not_contain.startswith('regex:'):
                pattern = re.compile(must_not_contain.replace('regex:', ''))
                self.file_paths = [fname for fname in self.file_paths if not pattern.search(fname)]
            elif any(c in must_not_contain for c in '*?['):
                self.file_paths = [fname for fname in self.file_paths if not fnmatch.fnmatch(fname, must_not_contain)]
            else:
                self.file_paths = [fname for fname in self.file_paths if must_not_contain not in fname]

        # self.file_paths = self.filter_files(self.file_paths)

        self._axis_name = ['antenna', 'frame', 'channel']
        self._label_dim = 55
        self._name = "XRF55Dataset"
        self._image_shape = resolution 
        actual_csi_shape = self._image_shape

        self.idx_list = np.arange(len(self.file_paths))
        self.live_idx, self.dead_idx = self.split_datasets(self.idx_list)

        # Load one sample to determine the shape of the csi data.
        csi_data, _, _, _ = _process_xrf55_file(self.file_paths[self.live_idx[0]], 3, self._label_dim)

        self._image_shape = csi_data.shape
        self._file_image_shape = self._image_shape

        if self.transpose is not None:
            actual_csi_shape = self._image_shape
            img_ndim = len(actual_csi_shape)
            # Assumes transpose is a permutation of the first N axes that are being transposed
            transpose_axes = self.transpose + tuple(range(len(self.transpose), img_ndim))
            
            actual_csi_shape = [actual_csi_shape[i] for i in transpose_axes]
            self._axis_name = [self._axis_name[i] for i in transpose_axes]

        if not self.view_as_complex:
            if self.complex_merge_axis is not None:
                actual_csi_shape[self.complex_merge_axis] *= 2
                self._axis_name[self.complex_merge_axis] = 'complex/' + self._axis_name[self.complex_merge_axis]
            else:
                actual_csi_shape.append(2)
                self._axis_name.append('complex')
        
        self._image_shape = tuple(actual_csi_shape)
        # --- End of shape calculation logic ---


        self.transpose_collate_fn = TransposeCollateFn(self.transpose)
        self.complex_view_collate_fn = ComplexViewCollateFn(self.view_as_complex, self.complex_merge_axis)
        if self.noise_mean_alter_way:
            self.corruption_collate_fn = AlternativeCorruptionCollateFn(
                additive_noise_sigma=self.additive_noise_sigma,
                multiply_noise_sigma=self.multiply_noise_sigma,
                corruption_probability_per_image=self.corruption_probability_per_image,
                only_additive_noise=self.only_additive_noise,
                image_corruption_seed=self.image_corruption_seed,
                image_noise_seed=self.image_noise_seed,
            )
        else:
            self.corruption_collate_fn = CorruptionCollateFn(
                additive_noise_sigma=self.additive_noise_sigma,
                multiply_noise_sigma=self.multiply_noise_sigma,
                corruption_probability_per_image=self.corruption_probability_per_image,
                only_additive_noise=self.only_additive_noise,
                image_corruption_seed=self.image_corruption_seed,
                image_noise_seed=self.image_noise_seed,
            )
        self.noise_mean_collate_fn = NoiseMeanCollateFn(self.noise_mean_flag, self.noise_mean_alter_way)

    def split_antenna_axis(self, csi_data, antenna_num=3):
        # input csi_data shape : (frame, antenna*channel)
        # output csi_data shape : (frame, antenna, channel)

        csi_data = np.split(csi_data, axis=1, indices_or_sections=antenna_num)
        return np.stack(csi_data)


    def _normalize_data(self, csi_data, noise_sigma_data):

        # if self.sigma_norm:
        #     # Sigma Normalization
        #     csi_data /= np.maximum(noise_sigma_data, 1e-12)
        #     noise_sigma_data = np.ones_like(noise_sigma_data)
        # else:
        #     mean_sigma = np.sqrt(np.mean(noise_sigma_data**2))
        #     csi_data /= mean_sigma
        #     noise_sigma_data /= mean_sigma

        csi_data /= self.normalize_value
        noise_sigma_data /= self.normalize_value

        # fit noise_sigma_data shape to csi_data shape for later processing
        while len(csi_data.shape) > len(noise_sigma_data.shape):
             noise_sigma_data = np.expand_dims(noise_sigma_data, axis=-1)
        # noise_sigma_data : (frame, antenna, 1) - float

        return csi_data, noise_sigma_data

    def _apply_transpose(self, csi_data, noise_sigma_data, axis_name):
        if self.transpose is not None:
            # For WiDAR, label is 1D vector, so we don't transpose it.
            item = {'image': csi_data, 'label': np.array([]), 'sigma': noise_sigma_data, 'axis_name': axis_name}
            item = self.transpose_collate_fn([item])[0]
            csi_data = item['image']
            noise_sigma_data = item['sigma']
            axis_name = item['axis_name']
        return csi_data, noise_sigma_data, axis_name

    def _handle_complex_view(self, csi_data, noise_sigma_data, axis_name):
        # For WiDAR, label is 1D vector, so we don't process it as complex.
        item = {'image': csi_data, 'label': np.array([]), 'sigma': noise_sigma_data, 'axis_name': axis_name}
        item = self.complex_view_collate_fn([item])[0]
        csi_data = item['image']
        noise_sigma_data = item['sigma']
        axis_name = item['axis_name']
        return csi_data, noise_sigma_data, axis_name

    def _apply_corruption(self, csi_data, noise_sigma_data, idx):
        item = {'image': csi_data, 'sigma': noise_sigma_data, 'idx': idx}
        item = self.corruption_collate_fn([item])[0]
        return item['image'], item['sigma'], item['corruption_label']

    def flip_splits(self):
        tmp = self.live_idx
        self.live_idx = self.dead_idx
        self.dead_idx = tmp
    

    def filter_files(self, file_paths):
        filtered_paths = []
        
        must_have = [self.must_have] if isinstance(self.must_have, str) else self.must_have
        must_not_have = [self.must_not_have] if isinstance(self.must_not_have, str) else self.must_not_have
        for path in file_paths:
            os.path.normpath(path)
            if must_have and not any(fnmatch.fnmatch(path, f'*{feature}*') for feature in must_have):
                continue
            if must_not_have and any(fnmatch.fnmatch(path, f'*{feature}*') for feature in must_not_have):
                continue
            filtered_paths.append(path)
        return filtered_paths
    
    def split_datasets(self, file_paths):
        split_shuffler = np.random.default_rng(seed=self.split_seed)
        split_shuffler.shuffle(file_paths)
        split_index = int(self.split_ratio * len(file_paths))
        return file_paths[:split_index], file_paths[split_index:]
    
    def get_gesture_num_from_path(self, file_path):
        # Extract gesture label from file path (assuming format includes '-gestureX-')
        base_name = os.path.basename(file_path)
        parts = base_name.split('-')
        for part in parts:
            if part.startswith('gesture'):
                return int(part.replace('gesture', ''))
        return None  # Return None if no gesture label is found
        
    def __len__(self):
        return len(self.live_idx)
    
    def __getitem__(self, idx):
        true_idx = self.live_idx[idx]
        file_path = self.file_paths[true_idx]

        csi_data, noise_sigma_data, _, label_data = _process_xrf55_file(file_path, 3, self._label_dim)
        if not self.use_labels:
            label_data = np.zeros_like(label_data)
        assert csi_data.shape == self._file_image_shape, f"Error: Expected CSI data shape {self._file_image_shape}, but got {csi_data.shape} for file {file_path}"
        axis_name = self._axis_name.copy()

        # 1. Normalize
        csi_data, noise_sigma_data = self._normalize_data(csi_data, noise_sigma_data)
        
        # 2. Transpose
        csi_data, noise_sigma_data, axis_name = self._apply_transpose(csi_data, noise_sigma_data, axis_name)

        # 3. Handle complex view
        csi_data, noise_sigma_data, axis_name = self._handle_complex_view(csi_data, noise_sigma_data, axis_name)

        # 4. Apply corruption
        csi_data, noise_sigma_data, corruption_label = self._apply_corruption(csi_data, noise_sigma_data, idx)

        # 5. Noise mean
        if self.noise_mean_flag:
            item = {'sigma': noise_sigma_data}
            item = self.noise_mean_collate_fn([item])[0]
            noise_sigma_data = item['sigma']

        if self.view_as_complex:
            dtype = np.complex64
        else:
            dtype = np.float32
        
        if self.complex_merge_axis is None:
            return_complex_merge_axis = -1
        else:
            return_complex_merge_axis = self.complex_merge_axis

        return {
            'image': csi_data.astype(dtype),
            'sigma': noise_sigma_data.astype(np.float32),
            'label': label_data.astype(np.float32),
            'filename': file_path,
            'idx': idx,
            'corruption_label': corruption_label,
            'additive_noise_sigma': self.additive_noise_sigma,
            'complex_merge_axis': return_complex_merge_axis,
            'axis_name': axis_name,
        }
    
    @property
    def __name__(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._image_shape)

    @property
    def resolution(self):
        # Returns the shape of the spatial dimensions (all dimensions except the first one).
        return self.image_shape[1:]

    @property
    def name(self):
        return self._name

    @property
    def label_dim(self):
        return self._label_dim

    @property
    def has_labels(self):
        return True
    
    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]
    
    @property
    def calculate_normalized_value(self):
        var_sum = 0.0
        var_count = 0

        noise_var_sum = 0.0
        noise_var_count = 0

        for idx in tqdm(self.live_idx, desc="Calculating normalized value"):
            file_path = self.file_paths[idx]
            csi_data, noise_sigma_data, _, _ = _process_xrf55_file(file_path, 3, self._label_dim)
            csi_data, noise_sigma_data = self._normalize_data(csi_data, noise_sigma_data)
            var_sum += np.sum(np.sqrt((csi_data * np.conj(csi_data)).real.flatten()))
            var_count += csi_data.flatten().size

            noise_var_sum += (np.sum((noise_sigma_data**2).flatten()))
            noise_var_count += noise_sigma_data.flatten().size

        return (var_sum / var_count), np.sqrt(noise_var_sum / noise_var_count)



if __name__ == "__main__":
    import sys
    # path = sys.argv[1] if len(sys.argv) > 1 else "../data/XRF55_noise_calculated/Scene1"
    # print(path)
    # dataset = XRF55Dataset(path=path, view_as_complex=True, dataset_keep_percentage=1.0, normalize_value=9.114174, must_not_contain=r"regex:\d{2}_\d{2}_(1[5-9]|20)_\d{2}\.npz$")

    # path = sys.argv[1] if len(sys.argv) > 1 else "../data/widar_preprocess/256_recal_noise/train"
    # dataset = WiDARDataset(path=path, view_as_complex=True, dataset_keep_percentage=1.0, must_not_contain="-user5-")

    path = sys.argv[1] if len(sys.argv) > 1 else "../data/RENEW/updown_link_test/train"
    dataset = renewRfProcessedDataset(path=path, view_as_complex=True, dataset_keep_percentage=1.0)

    print(dataset.calculate_normalized_value)
