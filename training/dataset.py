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

class renewRfDataset(ambient_utils.dataset_utils.Dataset):
    def __init__(self, 
                 path,                   # Path to files.
                 resolution      = None, # Ensure specific resolution, None = highest available.
                 must_contain    = None, # Require filenames to contain this substring.
                 must_not_contain = None, # Require filenames to NOT contain this substring.
                 sigma: float = 0.1,     # ensured minimum currption sigma
                 utilize_remaining_frame = False,
                 view_as_complex = False,
                 complex_merge_axis = None | int,
                 transpose = None | tuple,
                 **super_kwargs):
        self.minimum_sigma = sigma

        self._path = path
        if isinstance(self._path, list):
            self._all_fnames = self._path
        elif os.path.isdir(self._path):
            self._all_fnames = {os.path.join(self._path, os.path.relpath(os.path.join(root, fname), start=self._path)) for root, _dirs, files in os.walk(self._path) for fname in files}
        else:
            raise IOError('Path must point to a directory or list of file paths')
        
        self._prefix_fname = set([fname.replace('.csi.npy', '').replace('.noise.npy', '') for fname in self._all_fnames])
        
        if must_contain is not None:
            self._prefix_fname = {fname for fname in self._prefix_fname if must_contain in fname}
        
        if must_not_contain is not None:
            self._prefix_fname = {fname for fname in self._prefix_fname if must_not_contain not in fname}

        self._prefix_fname = sorted(list(self._prefix_fname))

        self._resolution = resolution

        if type(self._resolution) is tuple or type(self._resolution) is list:
            self._frame_resolution = self._resolution[0]
            self._ant_resolution = self._resolution[1]
        elif type(self._resolution) is int:
            self._frame_resolution = self._resolution
            self._ant_resolution = self._resolution
        else:
            assert False, "resolution must be int or tuple/list of int"
            
        # data shape = (frame, user, antenna, channel)
        self._csi_raw_data_list = [np.load(fprefix+'.csi.npy', mmap_mode='r') for fprefix in self._prefix_fname]
        self._noise_raw_data_list = [np.load(fprefix+'.noise.npy', mmap_mode='r') for fprefix in self._prefix_fname]
        # print(self._csi_raw_data_list[0].shape)
        
        self.each_data_idx = []

        for idx, csi_data in enumerate(self._csi_raw_data_list):
            frame_shape = csi_data.shape[0]
            user_shape = csi_data.shape[1]
            ant_shape = csi_data.shape[2]

            assert csi_data.shape[:-1] == self._noise_raw_data_list[idx].shape, f"CSI and noise data shape mismatch for prefix {self._prefix_fname[idx]}"
            assert ant_shape % self._ant_resolution == 0, f"Antenna dimension {ant_shape} is not divisible by ant_resolution {self._ant_resolution}"

            for fidx in range(0, frame_shape, self._frame_resolution):
                if fidx + self._frame_resolution > frame_shape:
                    if utilize_remaining_frame:
                        fidx = frame_shape - self._frame_resolution
                    else:
                        continue

                for uidx in range(0, user_shape):
                    for aidx in range(0, ant_shape, self._ant_resolution):
                        self.each_data_idx.append((idx, fidx, uidx, aidx))

        self._view_as_complex = view_as_complex
        self._complex_merge_axis = complex_merge_axis
        self._transpose = transpose

        name = os.path.splitext(os.path.basename(self._path))[0]
        super().__init__(name=name, **super_kwargs)

    def __len__(self):
        return len(self.each_data_idx)

    def __getitem__(self, idx):
        idx_tuple = self.each_data_idx[idx]
        csi_data = self._csi_raw_data_list[idx_tuple[0]][idx_tuple[1]: idx_tuple[1]+self._frame_resolution,
                                                         idx_tuple[2],
                                                         idx_tuple[3]: idx_tuple[3]+self._ant_resolution]
        noise_data = self._noise_raw_data_list[idx_tuple[0]][idx_tuple[1]: idx_tuple[1]+self._frame_resolution,
                                                             idx_tuple[2],
                                                             idx_tuple[3]: idx_tuple[3]+self._ant_resolution]        
        # csi_data : (frame, user, antenna, channel)
        # noise_data : (frame, user, antenna)

        if self._transpose is not None:
            assert noise_data.ndim == len(self._transpose), "noise_data ndim and transpose length mismatch"
            csi_data = np.transpose(csi_data, self._transpose + (3,))
            noise_data = np.transpose(noise_data, self._transpose)

        if not self._view_as_complex:
            csi_data = np.expand_dims(np.array(csi_data), -1)
            noise_data = np.expand_dims(np.array(noise_data), -1)

            csi_data = csi_data.view(np.float64)
            noise_data = noise_data.view(np.float64)

            if self._complex_merge_axis is not None:
                csi_data = np.concatenate((np.take(csi_data, 0, axis=-1),
                                           np.take(csi_data, 1, axis=-1)), axis=self._complex_merge_axis)
                noise_data = np.concatenate((np.take(noise_data, 0, axis=-1),
                                             np.take(noise_data, 1, axis=-1)), axis=self._complex_merge_axis)
        elif self._complex_merge_axis is not None:
            import warnings
            warnings.warn("complex_merge_axis is only applicable when view_as_complex is False")

        return csi_data, noise_data

if __name__ == "__main__":
    pass