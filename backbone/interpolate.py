# -*- coding: utf-8 -*-
"""
Module describing the Interpolation of the electrostatic potential
"""

import numpy as np
from numba import njit,prange 


@njit('int32[:,:](float64,float64)', cache=True, fastmath=True)
def rad_interpol_points(sampling, cutoff):
    """Internal function for the indices of the radial interpolation points."""
    margin = int(np.ceil(cutoff / sampling))
    indices = np.arange(-margin, margin + 1, dtype=np.int32)
    
    num_points = 0
    for i in indices:
        for j in indices:
            if i**2 + j**2 <= margin**2:
                num_points += 1

    disc_indices = np.zeros((num_points, 2), dtype=np.int32)
    idx = 0
    for i in indices:
        for j in indices:
            if i**2 + j**2 <= margin**2:
                disc_indices[idx, 0] = i
                disc_indices[idx, 1] = j
                idx += 1
    
    return disc_indices

@njit('void(float32[:,:,:],int32[:],int32[:,:],float64[:,:],'\
      'float32[:,:],float32[:,:],float32[:],float32[:])',
      nogil=True, parallel=True)
def interpolate_radial_functions(array,rle_encoding, disc_indices,
                                 positions, vr,dvdr, r, sampling):
    
    dt = np.log(r[-1] / r[0]) / (r.shape[0] - 1)
    r0 = r[0]
    
    for p in prange(rle_encoding.shape[0] - 1): # Thread safe loop
        for i in prange(rle_encoding[p], rle_encoding[p + 1]):
            for j in prange(disc_indices.shape[0]): # Thread safe loop
            
                k = int(round(positions[i, 0] / sampling[0])) + disc_indices[j, 0]
                m = int(round(positions[i, 1] / sampling[1])) + disc_indices[j, 1]

                if (k < array.shape[1]) & (m < array.shape[2]) & (k >= 0) & (m >= 0):
                    r_interp = np.sqrt((k * sampling[0] - positions[i, 0]) ** 2 +
                                       (m * sampling[1] - positions[i, 1]) ** 2)

                    idx = int(np.floor(np.log(r_interp / r0 + 1e-7) / dt))
                    
                    if idx < 0:
                        array[p, k, m] += vr[i,0]
                    elif idx < r.shape[0] - 1:
                        array[p, k, m] += vr[i,idx]+ (r_interp - r[idx]) * dvdr[i, idx]
