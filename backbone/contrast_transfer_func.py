# -*- coding: utf-8 -*-
"""
Module describing and determining the Contrast Transfer Function (CTF) and abberations.
"""
import numpy as np
import numba as nb 
from numba import njit
from tiling_multislice_func import fft2_convolve_sci



@njit(nogil=True,cache=True)
def scherzer_point_resolution(Cs: float, wavelength: float):
    return (wavelength** 3 * np.abs(Cs) / 6) ** (1 / 4)


#Cs is the spherical aberration
@njit(nogil=True,cache=True)
def scherzer_defocus(Cs,wavelength):
    return np.sign(Cs) * np.sqrt(3 / 2 * np.abs(Cs) * wavelength)


@nb.vectorize([nb.complex64(nb.float32), nb.complex128(nb.float64)])
def complex_exponential(x):
    return np.cos(x) + 1.j * np.sin(x)


def polar_coordinates(x, y):
    """Calculate a polar grid for a given Cartesian grid."""
    alpha = np.sqrt(x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2)
    phi = np.arctan2(y.reshape((1, -1)), x.reshape((-1, 1)))
    return alpha, phi


def scattering_angles(gpts, sampling, wavelength):
    kx, ky = tuple(np.fft.fftfreq(n, d).astype(np.float32) for n, d in zip(gpts, sampling))
    alpha, phi = polar_coordinates(kx * wavelength, ky * wavelength)
    return alpha, phi


def set_parameters(para, parameters: dict):
    #: Aliases for the most commonly used optical aberrations.
    polar_aliases = {'defocus': 'C10', 'astigmatism': 'C12', 'astigmatism_angle': 'phi12',
                     'coma': 'C21', 'coma_angle': 'phi21', 'Cs': 'C30','C5': 'C50'}
    
    for symbol, value in parameters.items():
        if symbol in para.keys():
            para[symbol] = value

        elif symbol == 'defocus':
            para[polar_aliases[symbol]] = -value

        elif symbol in polar_aliases.keys():
           para[polar_aliases[symbol]] = value

        else:
            continue

    return para


def evaluate_aberrations(p,alpha, phi,wavelength):
    phase_error = np.zeros(alpha.shape, dtype=np.float32)
    
    
    if any([p[symbol] != 0. for symbol in ('C10', 'C12', 'phi12')]):
        phase_error += (1 / 2 * (alpha**2) *(p['C10'] +p['C12'] * np.cos(2 * (phi - p['phi12']))))

    if any([p[symbol] != 0. for symbol in ('C21', 'phi21', 'C23', 'phi23')]):
        phase_error += (1 / 3 * (alpha**2) * alpha *
                 (p['C21'] * np.cos(phi - p['phi21']) +
                  p['C23'] * np.cos(3 * (phi - p['phi23']))))
        
    if any([p[symbol] != 0. for symbol in ('C30', 'C32', 'phi32', 'C34', 'phi34')]):
        phase_error += (1 / 4 * (alpha**2) ** 2 *
                  (p['C30'] +
                   p['C32'] * np.cos(2 * (phi - p['phi32'])) +
                   p['C34'] * np.cos(4 * (phi - p['phi34']))))
    
    if any([p[symbol] != 0. for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
           phase_error += (1 / 5 * (alpha**2) ** 2 * alpha *
                    (p['C41'] * np.cos((phi - p['phi41'])) +
                     p['C43'] * np.cos(3 * (phi - p['phi43'])) +
                     p['C45'] * np.cos(5 * (phi - p['phi45']))))

    if any([p[symbol] != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
            phase_error += (1 / 6 * (alpha**2) ** 3 *
                      (p['C50'] +
                       p['C52'] * np.cos(2 * (phi - p['phi52'])) +
                       p['C54'] * np.cos(4 * (phi - p['phi54'])) +
                       p['C56'] * np.cos(6 * (phi - p['phi56']))))  
    
    phase_error *= 2 * np.pi / wavelength
    aberration = complex_exponential(-phase_error)
    return aberration


def evaluate_aperture(alpha, phi, rolloff, cutoff):
    # Normalize cutoff if needed
    cutoff /= 1e3

    if rolloff > 0.:
        # Normalize rolloff if needed
        rolloff /= 1e3

        # Calculate the aperture using the formula
        array = 0.5 * (1 + np.cos(np.pi * (alpha - cutoff + rolloff) / rolloff))

        # Apply conditions to the aperture array
        array[alpha > cutoff] = 0.
        array = np.where(alpha > cutoff - rolloff, array, np.ones_like(alpha, dtype=np.float32))
    else:
        # If rolloff is zero or negative, set the array based on the cutoff
        array = np.array(alpha < cutoff).astype(np.float32)

    return array



def evaluate_spatial_envelope(p, alpha, phi,wavelength,angular_spread):
    dchi_dk = 2 * np.pi / wavelength * (
                (p['C12'] * np.cos(2. * (phi - p['phi12'])) + p['C10']) * alpha +
                (p['C23'] * np.cos(3. * (phi - p['phi23'])) +
                 p['C21'] * np.cos(1. * (phi - p['phi21']))) * alpha ** 2 +
                (p['C34'] * np.cos(4. * (phi - p['phi34'])) +
                 p['C32'] * np.cos(2. * (phi - p['phi32'])) + p['C30']) * alpha ** 3 +
                (p['C45'] * np.cos(5. * (phi - p['phi45'])) +
                 p['C43'] * np.cos(3. * (phi - p['phi43'])) +
                 p['C41'] * np.cos(1. * (phi - p['phi41']))) * alpha ** 4 +
                (p['C56'] * np.cos(6. * (phi - p['phi56'])) +
                 p['C54'] * np.cos(4. * (phi - p['phi54'])) +
                 p['C52'] * np.cos(2. * (phi - p['phi52'])) + p['C50']) * alpha ** 5)
            
    dchi_dphi = -2 * np.pi / wavelength * (
                1 / 2. * (2. * p['C12'] * np.sin(2. * (phi - p['phi12']))) * alpha +
                1 / 3. * (3. * p['C23'] * np.sin(3. * (phi - p['phi23'])) +
                          1. * p['C21'] * np.sin(1. * (phi - p['phi21']))) * alpha ** 2 +
                1 / 4. * (4. * p['C34'] * np.sin(4. * (phi - p['phi34'])) +
                          2. * p['C32'] * np.sin(2. * (phi - p['phi32']))) * alpha ** 3 +
    
                1 / 5. * (5. * p['C45'] * np.sin(5. * (phi - p['phi45'])) +
                          3. * p['C43'] * np.sin(3. * (phi - p['phi43'])) +
                          1. * p['C41'] * np.sin(1. * (phi - p['phi41']))) * alpha ** 4 +
                1 / 6. * (6. * p['C56'] * np.sin(6. * (phi - p['phi56'])) +
                          4. * p['C54'] * np.sin(4. * (phi - p['phi54'])) +
                          2. * p['C52'] * np.sin(2. * (phi - p['phi52']))) * alpha ** 5)
    
    return np.exp(-np.sign(angular_spread) * (angular_spread / 2 / 1000) ** 2 *
                  (dchi_dk ** 2 + dchi_dphi ** 2))


def CTF_evaluate(para,alpha,phi,wavel,semiangle_cutoff,rolloff,
                 focal_spread,angular_spread,gaussian_spread):
    
    kernel = evaluate_aberrations(para,alpha,phi,wavel)
    
    if semiangle_cutoff < np.inf:
        kernel *= evaluate_aperture(alpha,phi,rolloff,semiangle_cutoff)

    if focal_spread > 0.:
        kernel *= np.exp(- (.5 * np.pi /wavel * focal_spread * alpha ** 2) ** 2).astype(np.float32)
    
    if angular_spread > 0.:
        kernel *= evaluate_spatial_envelope(para, alpha, phi,wavel,angular_spread)    

    if gaussian_spread > 0.:
        kernel *= np.exp(- .5 * gaussian_spread ** 2 * alpha ** 2 / wavel ** 2)
        
    return kernel


def CTF_image(full_image,ctf_para,wavel,sampling,
              semiangle_cutoff: float = np.inf, rolloff: float = 2,
              focal_spread: float = 0.,angular_spread: float = 0.,
              gaussian_spread: float = 0.,):

    alpha,phi = scattering_angles(full_image.shape[1:], sampling, wavel)
    
    #: Symbols for the polar representation of all optical aberrations up to the fifth order.
    polar_symbols = ('C10', 'C12', 'phi12',
                     'C21', 'phi21', 'C23', 'phi23',
                     'C30', 'C32', 'phi32', 'C34', 'phi34',
                     'C41', 'phi41', 'C43', 'phi43', 'C45', 'phi45',
                     'C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')
     
    
    para =  dict(zip(polar_symbols, [0.] * len(polar_symbols))) 
    para = set_parameters(para, ctf_para)
    
    kernel = CTF_evaluate(para, alpha, phi, wavel,semiangle_cutoff,rolloff,
                          focal_spread,angular_spread,gaussian_spread)

    return fft2_convolve_sci(full_image.copy(),kernel)


