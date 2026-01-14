# -*- coding: utf-8 -*-
"""
Module for the tiling-based multilsice algorithm 
"""

import numpy as np
import numba as nb
import scipy


def get_mask( gpts, sampling=(1,1),cutoff = (2/3.,2/3.),rolloff = 0.1):
    # Unpack grid points and sampling rates
    kx, ky = tuple(np.fft.fftfreq(n, d).astype(np.float32) 
                   for n, d in zip(gpts, sampling))
    # Compute magnitude of the frequency vector
    k = np.sqrt(kx[:, None] ** 2 + ky[None] ** 2)
    
    # Compute cutoff frequency
    kcut = 1 / max(sampling) / 2 *max(cutoff)
    
    #root raised cosine window filter
    array = 0.5 * (1 + np.cos(np.pi * (k - kcut + rolloff) / rolloff))
    
    # Apply the low-pass filter
    array[k > kcut] = 0
    array = np.where(k > kcut - rolloff, array, 1)

    return array
    
@nb.vectorize([nb.complex64(nb.float32), nb.complex128(nb.float64)])
def complex_exp(x):
    return np.cos(x) + 1.j * np.sin(x)


def propagator(gpts, sampling, wavel, aperture, dt):
    # Unpack grid points and sampling rates
    freqx, freqy = [np.fft.fftfreq(n, d).astype(np.float32) 
                    for n, d in zip(gpts, sampling)]

    # Compute the propagation function
    propagation_func = (complex_exp(-(freqx ** 2)[:, None] * np.pi * wavel * dt)*
                        complex_exp(-(freqy ** 2)[None] * np.pi * wavel * dt))

    # Generate a mask for the aperture using the get_mask function
    mask = get_mask(gpts, sampling, aperture, 0.1)

    # Apply the mask to the propagation function
    propagation_func *= mask

    return propagation_func



def fft2_convolve_sci(array,kernel):
    array = scipy.fft.fftn(array,axes=(-1,-2), overwrite_x=True,workers=array.shape[0])
    array *= kernel
    array = scipy.fft.ifftn(array,axes=(-1,-2), overwrite_x=True,workers=array.shape[0])
    return array


def multislice_sci(tiles,sampling,wavel,aperature,thickness,kappa,sigma):
    """
    Multislice function for tiling-based Multislice algorithm parallel

    Parameters
    ----------
    tiles : 4D/2D float32 Array
        Potential tiles.
    sampling : float64
        angstrom per pixel.
    wavel : float64
        wavelenght of the electron beam.
    aperature : int32
        antialiasing aperture.
    thickness : float64
        slice thickness.
    kappa : float64
        potential parametrizations.
    sigma : float64
        interaction parameter.

    Returns
    -------
    wavefront : 4D/2D float32 Array
        scattered electron wave tiles.

    """
    wavefront =  np.ones((tiles.shape[0],tiles.shape[-2],
                          tiles.shape[-1]),dtype=np.complex64)
    
    # Create antialiasing mask for transmission function
    alias_mask = get_mask(tiles.shape[2:], (1, 1),(2/3.,2/3.))
    
    # Generate Propagator  
    prop = propagator(tiles.shape[2:],sampling,wavel,aperature,thickness)

    # Loop through all the slices
    for ti in range(tiles.shape[1]):
        
        # Transform the Potential into Transmission Function
        trans = tiles[:,ti]/kappa
        trans = complex_exp(sigma*trans)
        trans = fft2_convolve_sci(trans,alias_mask)    
        
        wavefront *= trans # Perform Transmission
        wavefront = fft2_convolve_sci(wavefront,prop) # Perform  Propagation
    
    return wavefront 


@nb.njit('complex64[:,:](complex64[:,:,:,:], int32[:],int32[:])')
def combine_tiles(wave,no_tiles,tile_size):
    combine = np.zeros((wave.shape[2]*no_tiles[0],
                        wave.shape[3]*no_tiles[1]),wave.dtype)
    for idx in range(no_tiles[0]*no_tiles[1]):
        i,j = idx // no_tiles[0] , idx % no_tiles[0]
        
        combine[tile_size[0]*i:tile_size[0]*(i+1),
                  tile_size[1]*j:tile_size[1]*(j+1)] = wave[i,j]
    return combine

