#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multislice simulations of CrN twin boundary
Figure 6
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from timeit import default_timer as timer


from ase.spacegroup import crystal
from ase.visualize import view
from ase.build import make_supercell

import sys
sys.path.append('backbone')

from xyz_reader import calculate_energy_constants
from Caching import Cache_original
from potential_creator import potential_slices,potential_tiles
from tiling_multislice_func import multislice_sci,combine_tiles
from contrast_transfer_func import CTF_image,scherzer_defocus
from diffraction_func import diffraction,fourier_offset,downsample 


def normalise(array):
    return ((array - np.min(array)) / (np.max(array) - np.min(array)))

###############################################################################
# specimen generation

crn = crystal(
    symbols=['Cr', 'N'],
    basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
    spacegroup=225,
    cellpar=[4.19, 4.19, 4.19, 90, 90, 90])


supercell_matrix = [[40, 0, 0], [0, 20, 0], [0, 0, 10]]

slab_1 = crn.copy()
slab_1.rotate(a=40, v=[0,0,1], center='COP', rotate_cell=False)
slab_1 = make_supercell(slab_1, supercell_matrix)


slab_2 = crn.copy()
slab_2.rotate(a=-40, v=[0,0,1], center='COP', rotate_cell=False)

slab_2 = make_supercell(slab_2, supercell_matrix)
slab_2.positions[:, 1] += slab_1.get_cell()[1,1]

# Combine the two slabs
combined_slab = slab_1 + slab_2
combined_slab.cell = np.max(combined_slab.positions,axis=0)
combined_slab.center(vacuum=4.19/2)
view(combined_slab)


###############################################################################
#  specimen parameters. 

const = calculate_energy_constants(300e3)# energy of the beam
const['image_size'] = np.array([2048,2048])
const['aperature'] = np.array([2/3]*2)

const['lattice'] = combined_slab.cell.lengths()
atoms = np.concatenate([combined_slab.copy().positions,
                        combined_slab.numbers.reshape((combined_slab.numbers.shape[0],1))],axis=1)


# specimen creation
const['trans_thickness'] = .1
const['sampling'] = const['lattice'][:2]/const['image_size']
const['angular'] =  1 /( const['lattice'][:2]) *const['wavel'] * 1e3
const['no_slices'] = int(np.ceil(const['lattice'][2]/const['trans_thickness']))
const['prop_thickness'] = const['lattice'][2] / const['no_slices'] 

const['offset'] = np.array([[0],[0]])
const['extent'] = np.array([[0,(const['image_size'][0]-1)*const['sampling'][0]],
                            [0,(const['image_size'][1]-1)*const['sampling'][1]]])
const['extent'] += const['offset']

print('Cell: {}'.format(const['lattice']))
print('No Slices: {}'.format(const['no_slices']))
print('Sampling: {}'.format(const['sampling']))


###############################################################################

# CTF aberrations
ctf_para = {'defocus': -98.7, 'Cs': -33e-6*1e10, 'semiangle': 60, 'focal':0,
            'angular':0, 'gauss':0}
ctf_para['defocus'] = scherzer_defocus(ctf_para['Cs'],const['wavel'])

# Diffraction Parameters
diff_para = {'mode' : 'valid'}
diff_para['down_size'] = np.array(downsample(const['image_size'], diff_para['mode'],
                                             const['lattice'][:2],const['aperature'], 
                                             const['wavel']))

diff_para['down_aperture'] = const['aperature'] * const['image_size'] / diff_para['down_size']
diff_para['down_samp'] = const['lattice'][:2]/ diff_para['down_size']
diff_para['down_offset'] = fourier_offset(diff_para['down_size'],diff_para['down_samp'])
diff_para['down_offset'] *= const['wavel']*1e3
diff_para['down_extent'] = np.array([[0,(diff_para['down_size'][0]-1)*const['angular'][0]],
                                     [0,(diff_para['down_size'][1]-1)*const['angular'][1]]])    
diff_para['down_extent'] += diff_para['down_offset']
diff_para['sampling'] = np.abs(np.sum(diff_para['down_extent']/diff_para['down_size'],axis=0))


###############################################################################
# conventional TEM multislice

conv_pot = np.zeros((const['no_slices'],const['image_size'][0],
                     const['image_size'][1]),np.float32)

cache = Cache_original(4096,nb.typed.Dict.empty(key_type=nb.types.int64,
                                        value_type=nb.types.float64[:]))

# determine potential slices
start = timer()
potential_slices(atoms,conv_pot,const['trans_thickness'],
                 const['sampling'],const['lattice'],cache)
end = timer()
print('Time Taken for Potential Slicing:{}'.format(end-start))
cache.clear()

# determine electron wave scattering via multislice
start = timer()
conv_exit = multislice_sci(conv_pot[None],const['sampling'],const['wavel'],
                           const['aperature'],const['prop_thickness'],
                           const['kappa'],const['sigma'])
end = timer()
print('Time Taken for Standard Multislice:{}'.format(end-start))


conv_exit = CTF_image(conv_exit,ctf_para, const['wavel'],
                      const['sampling'],ctf_para['semiangle'],2,
                      ctf_para['focal'],ctf_para['angular'],ctf_para['gauss'])

conv_int = normalise(np.abs(conv_exit[0])**2)

conv_ED = diffraction(conv_exit,const['image_size'],const['lattice'][:2],
                      const['wavel'],const['aperature'],diff_para['mode'])[0]


###############################################################################
# tiling parameters


const['no_tiles'] = np.array([2,2,1],np.int32) # No.tiles
const['tile_size'] = (const['image_size']/const['no_tiles'][:2]).astype(np.int32)


const['buffer'] = (const['image_size']/16).astype(np.int32) # buffer region
print('Buffer size: {}'.format(const['buffer']))

const['sub_lattice'] = const['lattice']/const['no_tiles'] # tile volume
const['buf_rad'] = const['buffer']*const['sampling'] # physical buffer region 
const['buf_rad'] = np.append(const['buf_rad'],0)

diff_para['down_buffer'] = (const['buffer']/const['image_size'])*diff_para['down_size']
diff_para['down_buffer'] = np.round(diff_para['down_buffer']).astype(np.int32)
diff_para['down_buff_rad']  = const['wavel']*1e3/(diff_para['down_buffer']*diff_para['down_samp'])



###############################################################################

# TEM tiling-based multislice


tiles_pot = np.zeros((np.prod(const['no_tiles'][:2]),const['no_slices'],
                      const['tile_size'][0]+2*const['buffer'][0],
                      const['tile_size'][1]+2*const['buffer'][1],),np.float32)

cache = Cache_original(4096,nb.typed.Dict.empty(key_type=nb.types.int64,
                                value_type=nb.types.float64[:]))

# determine potential tiles
start = timer()
potential_tiles(tiles_pot,const['no_tiles'],atoms,const['buf_rad'],
                const['sub_lattice'],const['trans_thickness'],
                const['sampling'],cache)
end = timer()
print('Time Taken for Potential Tiling:{}'.format(end - start))
cache.clear()

# determine electron wave scattering via tiling-based multislice
start = timer()
tiles_exit = multislice_sci(tiles_pot,const['sampling'],const['wavel'],
                            const['aperature'], const['prop_thickness'],
                            const['kappa'],const['sigma'])
end = timer()
print('Time Taken for Tiling Multislice:{}'.format(end - start))

# combine wave tiles
start = timer()
tiles_exit = tiles_exit[:,const['buffer'][0]:-const['buffer'][0],
                          const['buffer'][1]:-const['buffer'][1]]

tiles_exit = tiles_exit.reshape((const['no_tiles'][0],const['no_tiles'][1],
                                 const['tile_size'][0],const['tile_size'][1]))
tiles_exit = combine_tiles(tiles_exit,const['no_tiles'],const['tile_size'])[None]

end = timer()
print('Time Taken for Recombining:{}'.format(end - start))


# add abberations

tiles_exit = CTF_image(tiles_exit,ctf_para, const['wavel'],const['sampling'],
                        ctf_para['semiangle'],2
                        ,ctf_para['focal'],ctf_para['angular'],ctf_para['gauss'])

tiles_int = normalise(np.abs(tiles_exit[0])**2)
tiles_ED = diffraction(tiles_exit,const['image_size'],const['lattice'][:2],
                        const['wavel'],const['aperature'],diff_para['mode'])[0]


###############################################################################
# Plotting 


x_coords = np.arange(const['image_size'][0]/2,const['image_size'][0]*3/4,).astype(int)-256
y_coords = np.linspace(const['image_size'][1]/2,const['image_size'][1]/2, 
                       int(const['image_size'][1]/4)).astype(int)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Create 1 row, 3 column subplot

# Shared parameters
xlims = (const['lattice'][0]/2 - const['buf_rad'][0]*2,
         const['lattice'][0]/2 + const['buf_rad'][0]*2)
ylims = (const['lattice'][1]/2 - const['buf_rad'][1]*2,
         const['lattice'][1]/2 + const['buf_rad'][1]*2)
extent = const['extent'].flatten()

ax = axes[0]
im = ax.imshow(conv_int, extent=extent, cmap='gray', origin='lower',
               vmin=np.min(conv_int), vmax=np.max(conv_int), interpolation='nearest')
ax.plot(x_coords * const['sampling'][0], y_coords * const['sampling'][1], 'b--')
fig.colorbar(im, ax=ax, shrink=0.7)
ax.set_title('(a) Conv BF', loc='left', fontsize=12)
ax.set_xlabel('x (\u212b)', fontsize=12)
ax.set_ylabel('y (\u212b)', fontsize=12)
ax.tick_params(labelsize=12)
ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax = axes[1]
im = ax.imshow(tiles_int, extent=extent, cmap='gray', origin='lower',
               vmin=np.min(conv_int), vmax=np.max(conv_int), interpolation='nearest')
ax.plot(x_coords * const['sampling'][0], y_coords * const['sampling'][1], 'r-.')
fig.colorbar(im, ax=ax, shrink=0.7)
ax.set_title('(b) Tiling BF', loc='left', fontsize=12)
ax.set_xlabel('x (\u212b)', fontsize=12)
ax.set_ylabel('y (\u212b)', fontsize=12)
ax.tick_params(labelsize=12)
ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax = axes[2]
ax.plot(x_coords * const['sampling'][0], conv_int[y_coords, x_coords], 'b--', label='Conv BF')
ax.plot(x_coords * const['sampling'][0], tiles_int[y_coords, x_coords], 'r-.', label='Tiling BF')
ax.set_title('(c) BF (comparison)', loc='left', fontsize=12)
ax.set_xlabel('x (\u212b)', fontsize=12)
ax.set_ylabel('Intensity', fontsize=12)
ax.tick_params(labelsize=12)
ax.legend()

ax = axes[3]
ax.plot(x_coords * const['sampling'][0], (conv_int-tiles_int)[y_coords, x_coords], 
        'g-', label='Diff BF')
ax.set_title('(d) BF (Difference)', loc='left', fontsize=12)
ax.set_xlabel('x (\u212b)', fontsize=12)
ax.set_ylabel('Intensity', fontsize=12)
ax.tick_params(labelsize=12)
ax.legend()
ax.ticklabel_format(style="plain",  axis="y")

plt.tight_layout()
plt.show()

