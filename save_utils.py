# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Save/load utility entry point (re-exports from torch_utils.save_utils)."""

from torch_utils.save_utils import (
    is_zlib_compressed,
    decompress_if_compressed,
    save_pkl,
    save_pt,
    save_file,
    load_pkl,
    load_pt,
    load_file,
)

__all__ = [
    'is_zlib_compressed',
    'decompress_if_compressed',
    'save_pkl',
    'save_pt',
    'save_file',
    'load_pkl',
    'load_pt',
    'load_file',
]
