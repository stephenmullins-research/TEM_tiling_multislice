#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 code for Graphing Fresnel diffraction
"""

import numpy as np
import matplotlib.pyplot as plt

def fresnel_integrals(u):
    # Numerical approximation of Fresnel integrals
    du = 0.001
    t = np.arange(0, np.abs(u) + du, du)

    Cos = np.trapz(np.cos(np.pi * t ** 2 / 2), t)
    Sine = np.trapz(np.sin(np.pi * t ** 2 / 2), t)

    if u < 0:
        Cos = -Cos
        Sine = -Sine

    return Cos, Sine

def fresnel_edge_diffraction(x, wavelength, z_prop):
    # Calculate the Fresnel number for each point
    w = np.sqrt(2 / (wavelength * z_prop)) * x

    # Fresnel integrals
    Cos = np.zeros(w.shape)
    Sine = np.zeros(w.shape)

    for i, u in enumerate(w):
        Cos[i], Sine[i] = fresnel_integrals(-u)

    # Intensity using the Fresnel integrals
    Intensity = 0.5 * ((0.5 - Cos) ** 2 + (0.5 - Sine) ** 2)
    Intensity /=np.max(Intensity)
    return Intensity,w


# Parameters

wavelength = 1.968748889772767e-12 # @300keV
# wavelength = 2.5e-12  # @200keV
# wavelength = 3.7e-12  # @100keV
# wavelength = 4.18e-12  # @80keV

N = 100
z_prop = 1e-10
position = np.linspace(-0.1e-9, 1e-9, 1000)  # Range in meters

# Calculate the Fresnel edge diffraction pattern
intensity,omega = fresnel_edge_diffraction(position, wavelength, N*z_prop)




# Calculate intensity modulation
local_maxima = (np.diff(np.sign(np.diff(intensity))) < 0).nonzero()[0] + 1
local_minima = (np.diff(np.sign(np.diff(intensity))) > 0).nonzero()[0] + 1

intensity_maxima = intensity[local_maxima]
intensity_minima = intensity[local_minima]

modulation = (intensity_maxima - intensity_minima)/2
modulation /= (intensity_maxima + intensity_minima)/2

print("\n Troughs Omegas:\n", omega[local_minima])

sampling = 0.09765625e-10

buffer_test = np.ceil(omega[local_maxima]*np.sqrt(wavelength*N*z_prop/2)/sampling)
print(buffer_test)


fig = plt.figure(figsize=(12,6))
fig.add_subplot(1,2,1)
plt.plot(position*1e9, intensity,'g-',alpha=0.7)
plt.scatter(position[local_maxima][9]*1e9, intensity_maxima[9], s=60, c='b', marker='x',)
plt.scatter(position[local_minima][9]*1e9, intensity_minima[9], s=60, c='b', marker='x',)
plt.xlim(-0.05,0.8)
plt.xlabel('Distance from Edge (nm)')
plt.ylabel('Relative Intensity')
plt.title('(a) Fresnel Diffraction of Opaque Straight Edge')
plt.grid(True)

fig.add_subplot(1,2,2)
plt.plot(position[local_minima]*1e9,modulation,'g.-',alpha=0.5)
plt.scatter(position[local_minima][9]*1e9,modulation[9],s=60, c='b', marker='x',)
plt.xlim(0.15,0.8)
plt.ylabel('Relative Intensity Modulation')
plt.xlabel('Distance from Edge (nm)')
plt.title('(b) Fresnel Diffraction Intensity Modulation')
plt.grid(True)

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(12,6))
fig.add_subplot(1,2,1)
plt.plot(omega,intensity,'g-',alpha=0.7)
plt.scatter(omega[local_maxima][9], intensity_maxima[9], s=60, c='b', marker='x',)
plt.scatter(omega[local_minima][9], intensity_minima[9], s=60, c='b', marker='x',)
plt.ylabel('Relative Intensity')
plt.xlabel('Fresnel Number')
plt.title('Fresnel Diffraction of Straight Edge')
plt.grid(True)

fig.add_subplot(1,2,2)
plt.plot(omega[local_minima],modulation,'g.-',alpha=0.5)
plt.scatter(omega[local_minima][9],modulation[9],s=60, c='b', marker='x',)
plt.ylabel('Relative Intensity Modulation')
plt.xlabel('Fresnel Number')
plt.title('Fresnel Diffraction Intensity Modulation')
plt.grid(True)

plt.tight_layout()
plt.show()
