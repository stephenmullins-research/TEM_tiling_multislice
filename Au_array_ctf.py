# -*- coding: utf-8 -*-

"""
Figure 9 aberrations and image post processing pf Gold nanopartilce array
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import sys
sys.path.append('backbone')
from xyz_reader import calculate_energy_constants
from contrast_transfer_func import CTF_image 


def normalise(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


const = calculate_energy_constants(300e3)# energy of the beam
wave = np.load('Au_exit_wave_[8].npy')
wave_int = np.abs(wave)**2



fig = plt.figure(figsize=(8,8))
plt.imshow(wave_int[0],
           cmap='gray',origin='lower', interpolation='nearest')
scalebar = ScaleBar(0.244/10,'nm', box_alpha=0,
                    location='lower right',color='w')
plt.gca().add_artist(scalebar)
plt.colorbar(shrink=0.75)
plt.tight_layout()
plt.show()



ctf_para = {'defocus': 1000, 'Cs': -7e-6 * 1e10}
ctf_para['semiangle'] = 36#np.inf
ctf_para['focal'], ctf_para['angular'] ,ctf_para['gaussian'] = 60, 0, 10

wave_ctf = CTF_image(wave.copy(),ctf_para, const['wavel'],
                     np.array([0.244,0.244]),ctf_para['semiangle'],2,
                      ctf_para['focal'],ctf_para['angular'],ctf_para['gaussian'])

fig = plt.figure(figsize=(6,6))
plt.imshow(normalise(np.abs(wave_ctf[0])**2),
           cmap='gray',origin='lower', interpolation='nearest')
scalebar = ScaleBar(0.244/10,'nm', box_alpha=0,
                    location='lower left',color='r')
plt.gca().add_artist(scalebar)
plt.colorbar(shrink=0.75)

plt.xticks(np.arange(0,8193,1024),fontsize=12)
plt.yticks(np.arange(0,8193,1024),fontsize=12)

plt.tight_layout()
plt.show()
plt.savefig('Au_array_plot.png', dpi = 300,pad_inches=0.1,bbox_inches='tight')

