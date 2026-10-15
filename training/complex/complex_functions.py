#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: spopoff
"""
import torch
from torch.nn.functional import relu, max_pool2d, max_pool3d, avg_pool2d, avg_pool1d, dropout, dropout2d, dropout3d, interpolate, sigmoid, tanh, leaky_relu

def _is_2_channel(x):
    return not torch.is_complex(x) and isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[1] == 2

def _to_complex(x):
    if _is_2_channel(x):
        return torch.complex(x[:, 0], x[:, 1]), True
    return x, False

def _to_output(res, was_2_channel):
    if was_2_channel and torch.is_complex(res):
        return torch.stack([res.real, res.imag], dim=1)
    return res

def complex_matmul(A, B):
    '''
        Performs the matrix product between two complex matrices
    '''
    A_c, is_2ch_A = _to_complex(A)
    B_c, is_2ch_B = _to_complex(B)

    outp_real = torch.matmul(A_c.real, B_c.real) - torch.matmul(A_c.imag, B_c.imag)
    outp_imag = torch.matmul(A_c.real, B_c.imag) + torch.matmul(A_c.imag, B_c.real)
    
    res = outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)
    return _to_output(res, is_2ch_A or is_2ch_B)

def complex_avg_pool1d(input, *args, **kwargs):
    inp, is_2ch = _to_complex(input)
    absolute_value_real = avg_pool1d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool1d(inp.imag, *args, **kwargs)    

    res = absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(torch.complex64)
    return _to_output(res, is_2ch)

def complex_avg_pool2d(input, *args, **kwargs):
    '''
    Perform complex average pooling.
    '''    
    inp, is_2ch = _to_complex(input)
    absolute_value_real = avg_pool2d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(inp.imag, *args, **kwargs)    
    
    res = absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(torch.complex64)
    return _to_output(res, is_2ch)

def complex_normalize(input):
    '''
    Perform complex normalization
    '''
    inp, is_2ch = _to_complex(input)
    real_value, imag_value = inp.real, inp.imag
    real_norm = (real_value - real_value.mean()) / real_value.std()
    imag_norm = (imag_value - imag_value.mean()) / imag_value.std()
    
    res = real_norm.type(torch.complex64) + 1j * imag_norm.type(torch.complex64)
    return _to_output(res, is_2ch)

def complex_leaky_relu(input, negative_slope):
    inp, is_2ch = _to_complex(input)
    res = leaky_relu(inp.real, negative_slope).type(torch.complex64) + 1j * leaky_relu(inp.imag, negative_slope).type(torch.complex64)
    return _to_output(res, is_2ch)

def complex_relu(input):
    inp, is_2ch = _to_complex(input)
    res = relu(inp.real).type(torch.complex64) + 1j * relu(inp.imag).type(torch.complex64)
    return _to_output(res, is_2ch)

def complex_sigmoid(input):
    inp, is_2ch = _to_complex(input)
    res = sigmoid(inp.real).type(torch.complex64) + 1j * sigmoid(inp.imag).type(torch.complex64)
    return _to_output(res, is_2ch)

def complex_tanh(input):
    inp, is_2ch = _to_complex(input)
    res = tanh(inp.real).type(torch.complex64) + 1j * tanh(inp.imag).type(torch.complex64)
    return _to_output(res, is_2ch)

def complex_opposite(input):
    inp, is_2ch = _to_complex(input)
    res = -(inp.real).type(torch.complex64) + 1j * (-(inp.imag).type(torch.complex64))
    return _to_output(res, is_2ch)

def complex_stack(input, dim):
    if len(input) > 0 and _is_2_channel(input[0]):
        actual_dim = dim + 1 if dim >= 0 else dim
        return torch.stack(input, dim=actual_dim)
    input_real = [x.real for x in input]
    input_imag = [x.imag for x in input]
    return torch.stack(input_real, dim).type(torch.complex64) + 1j * torch.stack(input_imag, dim).type(torch.complex64)

def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output

def _retrieve_elements_from_indices3d(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-3)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-3)).view_as(indices)
    return output

def complex_upsample(input, size=None, scale_factor=None, mode='nearest',
                             align_corners=None, recompute_scale_factor=None):
    '''
        Performs upsampling by separately interpolating the real and imaginary part and recombining
    '''
    inp, is_2ch = _to_complex(input)
    outp_real = interpolate(inp.real, size=size, scale_factor=scale_factor, mode=mode,
                            align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    outp_imag = interpolate(inp.imag, size=size, scale_factor=scale_factor, mode=mode,
                            align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    
    res = outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)
    return _to_output(res, is_2ch)

def complex_upsample2(input, size=None, scale_factor=None, mode='nearest',
                             align_corners=None, recompute_scale_factor=None):
    '''
        Performs upsampling by separately interpolating the amplitude and phase part and recombining
    '''
    inp, is_2ch = _to_complex(input)
    outp_abs = interpolate(inp.abs(), size=size, scale_factor=scale_factor, mode=mode,
                           align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    angle = torch.atan2(inp.imag, inp.real)
    outp_angle = interpolate(angle, size=size, scale_factor=scale_factor, mode=mode,
                             align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    
    res = outp_abs * (torch.cos(outp_angle).type(torch.complex64) + 1j * torch.sin(outp_angle).type(torch.complex64))
    return _to_output(res, is_2ch)


def complex_max_pool2d(input, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    inp, is_2ch = _to_complex(input)
    absolute_value, indices = max_pool2d(
        inp.abs(), 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        dilation=dilation,
        ceil_mode=ceil_mode, 
        return_indices=True
    )
    absolute_value = absolute_value.type(torch.complex64)
    angle = torch.atan2(inp.imag, inp.real)
    angle = _retrieve_elements_from_indices(angle, indices)
    res = absolute_value * (torch.cos(angle).type(torch.complex64) + 1j * torch.sin(angle).type(torch.complex64))
    return _to_output(res, is_2ch)

def complex_max_pool3d(input, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    inp, is_2ch = _to_complex(input)
    absolute_value, indices = max_pool3d(
        inp.abs(), 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        dilation=dilation,
        ceil_mode=ceil_mode, 
        return_indices=True
    )
    absolute_value = absolute_value.type(torch.complex64)
    angle = torch.atan2(inp.imag, inp.real)
    angle = _retrieve_elements_from_indices3d(angle, indices)
    res = absolute_value * (torch.cos(angle).type(torch.complex64) + 1j * torch.sin(angle).type(torch.complex64))
    return _to_output(res, is_2ch)


def complex_adaptive_avg_pool3d(input, kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):
    inp, is_2ch = _to_complex(input)
    absolute_value, indices = max_pool3d(
        inp.abs(), 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        dilation=dilation,
        ceil_mode=ceil_mode, 
        return_indices=True
    )
    absolute_value = absolute_value.type(torch.complex64)
    angle = torch.atan2(inp.imag, inp.real)
    angle = _retrieve_elements_from_indices(angle, indices)
    res = absolute_value * (torch.cos(angle).type(torch.complex64) + 1j * torch.sin(angle).type(torch.complex64))
    return _to_output(res, is_2ch)


def complex_dropout(input, p=0.5, training=True):
    if _is_2_channel(input):
        if p == 0.0 or not training:
            return input
        mask_shape = list(input.shape)
        mask_shape[1] = 1
        mask = torch.ones(mask_shape, dtype=torch.float32, device=input.device)
        mask = dropout(mask, p, training) * (1.0 / (1.0 - p))
        return mask.to(input.dtype) * input
    device = input.device
    mask = torch.ones(*input.shape, dtype=torch.float32, device=device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask = mask.type(input.dtype)
    return mask * input


def complex_dropout2d(input, p=0.5, training=True):
    if _is_2_channel(input):
        if p == 0.0 or not training:
            return input
        mask_shape = list(input.shape)
        mask_shape[1] = 1
        mask = torch.ones(mask_shape, dtype=torch.float32, device=input.device)
        mask = dropout2d(mask, p, training) * (1.0 / (1.0 - p))
        return mask.to(input.dtype) * input
    device = input.device
    mask = torch.ones(*input.shape, dtype=torch.float32, device=device)
    mask = dropout2d(mask, p, training) * 1 / (1 - p)
    mask = mask.type(input.dtype)
    return mask * input

def complex_dropout3d(input, p=0.5, training=True):
    if _is_2_channel(input):
        if p == 0.0 or not training:
            return input
        mask_shape = list(input.shape)
        mask_shape[1] = 1
        mask = torch.ones(mask_shape, dtype=torch.float32, device=input.device)
        mask = dropout3d(mask, p, training) * (1.0 / (1.0 - p))
        return mask.to(input.dtype) * input
    device = input.device
    mask = torch.ones(*input.shape, dtype=torch.float32, device=device)
    mask = dropout3d(mask, p, training) * 1 / (1 - p)
    mask = mask.type(input.dtype)
    return mask * input
