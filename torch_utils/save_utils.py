# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Unified model saving and loading utilities with zlib compression support."""

import os
import io
import zlib
import pickle
from typing import Any, Optional, Union
import torch
import dnnlib


def is_zlib_compressed(data: bytes) -> bool:
    """Check if the given bytes represent a zlib-compressed data stream.
    
    A valid zlib stream starts with a 2-byte header (CMF and FLG) where:
    - (CMF * 256 + FLG) % 31 == 0 (header checksum)
    - CMF & 0x0F == 8 (Deflate compression method)
    - (CMF >> 4) <= 7 (window size exponent <= 32K)
    We also safely probe decompression of the initial chunk to eliminate false positives.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) < 2:
        return False
    cmf = data[0]
    flg = data[1]
    if (cmf * 256 + flg) % 31 != 0:
        return False
    if (cmf & 0x0F) != 8:
        return False
    if (cmf >> 4) > 7:
        return False
    try:
        d = zlib.decompressobj()
        d.decompress(bytes(data[:min(len(data), 1024)]))
        return True
    except Exception:
        return False


def decompress_if_compressed(data: bytes) -> bytes:
    """Decompresses zlib-compressed bytes if compressed, otherwise returns original bytes."""
    if is_zlib_compressed(data):
        try:
            return zlib.decompress(data)
        except Exception:
            return data
    return data


def save_pkl(obj: Any, path: Union[str, os.PathLike], compress: bool = True) -> None:
    """Save an object as a pickle file, optionally compressed with zlib.
    
    Args:
        obj: The python object to serialize.
        path: Destination file path.
        compress: Whether to compress the serialized bytes with zlib (default: True).
    """
    path = os.fspath(path)
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    data = buf.getvalue()
    if compress:
        data = zlib.compress(data)
    
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        
    with open(path, 'wb') as f:
        f.write(data)


def save_pt(obj: Any, path: Union[str, os.PathLike], compress: bool = True) -> None:
    """Save an object (model, state dict, etc.) as a PyTorch file, optionally compressed with zlib.
    
    Args:
        obj: The PyTorch object or dict to serialize.
        path: Destination file path.
        compress: Whether to compress the serialized bytes with zlib (default: True).
    """
    path = os.fspath(path)
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    if compress:
        data = zlib.compress(data)
        
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        
    with open(path, 'wb') as f:
        f.write(data)


def save_file(obj: Any, path: Union[str, os.PathLike], compress: bool = True, format: Optional[str] = None) -> None:
    """Unified save function. Automatically chooses serialization method by format/extension.
    
    Args:
        obj: The object to serialize and save.
        path: Destination file path.
        compress: Whether to compress the serialized bytes with zlib (default: True).
        format: Explicit format ('pkl', 'pt', 'pth', 'raw'). If None, inferred from path extension.
    """
    path_str = str(path).lower()
    if format == 'pkl' or (format is None and (path_str.endswith('.pkl') or path_str.endswith('.pickle'))):
        save_pkl(obj, path, compress=compress)
    elif format in ('pt', 'pth') or (format is None and (path_str.endswith('.pt') or path_str.endswith('.pth'))):
        save_pt(obj, path, compress=compress)
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        data = bytes(obj)
        if compress:
            data = zlib.compress(data)
        dirname = os.path.dirname(os.path.abspath(path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)
    else:
        # Fallback to pickle
        save_pkl(obj, path, compress=compress)


def _read_raw_bytes(path_or_url_or_file: Any, verbose: bool = True) -> bytes:
    """Helper to read raw bytes from a file path, URL, file-like object, or bytes."""
    if isinstance(path_or_url_or_file, (bytes, bytearray, memoryview)):
        return bytes(path_or_url_or_file)
    if isinstance(path_or_url_or_file, (str, os.PathLike)):
        path_or_url = os.fspath(path_or_url_or_file)
        with dnnlib.util.open_url(path_or_url, verbose=verbose) as f:
            return f.read()
    if hasattr(path_or_url_or_file, 'read'):
        return path_or_url_or_file.read()
    raise TypeError(f"Unsupported input type for loading: {type(path_or_url_or_file)}")


def load_pkl(path_or_url_or_file: Any, verbose: bool = True) -> Any:
    """Load and deserialize a pickle object from a file path, URL, file stream, or bytes.
    Automatically detects and decompresses zlib compression if present.
    """
    raw_bytes = _read_raw_bytes(path_or_url_or_file, verbose=verbose)
    decompressed = decompress_if_compressed(raw_bytes)
    return pickle.loads(decompressed)


def load_pt(path_or_url_or_file: Any, map_location: Any = 'cpu', weights_only: bool = False, verbose: bool = True) -> Any:
    """Load and deserialize a PyTorch object from a file path, URL, file stream, or bytes.
    Automatically detects and decompresses zlib compression if present.
    """
    raw_bytes = _read_raw_bytes(path_or_url_or_file, verbose=verbose)
    decompressed = decompress_if_compressed(raw_bytes)
    return torch.load(io.BytesIO(decompressed), map_location=map_location, weights_only=weights_only)


def load_file(path_or_url_or_file: Any, map_location: Any = 'cpu', weights_only: bool = False, verbose: bool = True, format: Optional[str] = None) -> Any:
    """Unified load function. Reads from file path, URL, or stream, decompresses if zlib compressed,
    and deserializes according to format or file extension.
    """
    raw_bytes = _read_raw_bytes(path_or_url_or_file, verbose=verbose)
    decompressed = decompress_if_compressed(raw_bytes)
    
    path_str = str(path_or_url_or_file).lower() if isinstance(path_or_url_or_file, (str, os.PathLike)) else ""
    
    if format == 'pkl' or (format is None and (path_str.endswith('.pkl') or path_str.endswith('.pickle'))):
        return pickle.loads(decompressed)
    elif format in ('pt', 'pth') or (format is None and (path_str.endswith('.pt') or path_str.endswith('.pth'))):
        return torch.load(io.BytesIO(decompressed), map_location=map_location, weights_only=weights_only)
    else:
        # Auto-detect format from content
        if decompressed.startswith(b'PK\x03\x04'):
            # Standard PyTorch zip archive
            return torch.load(io.BytesIO(decompressed), map_location=map_location, weights_only=weights_only)
        try:
            return pickle.loads(decompressed)
        except Exception:
            return torch.load(io.BytesIO(decompressed), map_location=map_location, weights_only=weights_only)
