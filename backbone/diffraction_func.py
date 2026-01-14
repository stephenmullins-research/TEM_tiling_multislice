# -*- coding: utf-8 -*-
"""
Module for Diffraction Methods
"""

import numpy as np
import numba as nb
from numba import njit,prange
   

def _fft_interpolation_masks_1d(n1, n2):
    mask1 = np.zeros(n1, dtype=bool)
    mask2 = np.zeros(n2, dtype=bool)

    if n2 > n1:
        mask1[:] = True
    else:
        if n2 == 1:
            mask1[0] = True
        elif n2 % 2 == 0:
            mask1[:n2 // 2] = True
            mask1[-n2 // 2:] = True
        else:
            mask1[:n2 // 2 + 1] = True
            mask1[-n2 // 2 + 1:] = True

    if n1 > n2:
        mask2[:] = True
    else:
        if n1 == 1:
            mask2[0] = True
        elif n1 % 2 == 0:
            mask2[:n1 // 2] = True
            mask2[-n1 // 2:] = True
        else:
            mask2[:n1 // 2 + 1] = True
            mask2[-n1 // 2 + 1:] = True

    return mask1, mask2

def fft_interpolation_masks(shape1, shape2):
    mask1_1d = []
    mask2_1d = []

    for i, (n1, n2) in enumerate(zip(shape1, shape2)):
        m1, m2 = _fft_interpolation_masks_1d(n1, n2)

        s = [np.newaxis] * len(shape1)
        s[i] = slice(None)

        mask1_1d += [m1[tuple(s)]]
        mask2_1d += [m2[tuple(s)]]

    mask1 = mask1_1d[0]
    for m in mask1_1d[1:]:
        mask1 = mask1 * m

    mask2 = mask2_1d[0]
    for m in mask2_1d[1:]:
        mask2 = mask2 * m

    return mask1, mask2

def fft_crop(array, new_shape):

    mask_in, mask_out = fft_interpolation_masks(array.shape, new_shape)

    if len(new_shape) < len(array.shape):
        new_shape = array.shape[:-len(new_shape)] + new_shape

    new_array = np.zeros(new_shape, dtype=array.dtype)

    out_indices = np.where(mask_out)
    in_indices = np.where(mask_in)

    new_array[out_indices] = array[in_indices]
    return new_array


def downsample(gpts_full, max_angle,extent,aperature,wavel):

    if max_angle is None:
        gpts = gpts_full

    elif isinstance(max_angle, str):
        if max_angle == 'limit':
            cutoff_scattering_angle = np.floor(gpts_full*aperature-1e-12)
        elif max_angle == 'valid':
            cutoff_scattering_angle = np.floor(2*extent*min((aperature/(extent/gpts_full)/2 - 0.1))
                                               /np.sqrt(2)-1e-12)
        else:
            raise RuntimeError()

        gpts = np.ceil(cutoff_scattering_angle-1e-12).astype(np.int32)
    else:
        try:
            angular_sampling = 1 / extent * wavel * 1e3
            gpts = [int(2 * np.ceil(max_angle / d)) + 1 for n, d in zip(gpts_full, angular_sampling)]
        except:
            raise RuntimeError()

    return np.min([gpts, gpts_full],axis=0)



def diffraction(array,gpts_full,extent,wavel, aperature,max_angle='valid'):
    array = np.fft.fft2(array)
    gpts = downsample(gpts_full, max_angle,extent,aperature,wavel)
    if gpts[0] != gpts_full[0] and gpts[1] != gpts_full[1]:
        array = fft_crop(array, np.insert(gpts,0,1))
    diffr_array = np.fft.fftshift(np.abs(array)**2,axes=(-1,-2))
    return diffr_array


def fourier_offset(n: np.ndarray, d: np.ndarray) -> np.ndarray:

    offset = np.zeros_like(n, dtype=float)
    even_indices = n % 2 == 0
    odd_indices = ~even_indices

    offset[even_indices] = -1 / (2 * d[even_indices])
    offset[odd_indices] = -1 / (2 * d[odd_indices]) + 1 / (2 * d[odd_indices] * n[odd_indices])

    return offset


def remove_zeroth_order(diffraction_pattern,gpts,sampling,wavel,angular_radius=1):

    offset = np.array([-0.5/d if n % 2 == 0 else -0.5/d + 0.5/(d*n) for n, d in zip(gpts,sampling)])
    offset *= wavel* 1e3

    rel_spatial_freq = wavel * 1e3/(gpts*sampling)

    alpha_x = np.linspace(offset[0],gpts[0]*rel_spatial_freq[0]+offset[0], gpts[0], endpoint=False)
    alpha_y = np.linspace(offset[1], gpts[1]*rel_spatial_freq[1]+offset[1], gpts[1], endpoint=False)
    
    alpha_x, alpha_y = np.meshgrid(alpha_x, alpha_y, indexing='ij')

    alpha = alpha_x ** 2 + alpha_y ** 2
    block = alpha > angular_radius ** 2
    diffraction_pattern *= block
    
    return diffraction_pattern

