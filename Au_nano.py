# -*- coding: utf-8 -*-


"""
multislice simulations of gold nanoparticle 
Figure 4
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from abtem import Potential,PlaneWave,CTF
from ase.io import read

import sys
sys.path.append('backbone')

from xyz_reader import read_xyz_file,calculate_energy_constants
from Caching import Cache_original
from potential_creator import potential_slices,potential_tiles
from tiling_multislice_func import multislice_sci,combine_tiles
from contrast_transfer_func import CTF_image
from diffraction_func import diffraction,fourier_offset,downsample


def normalise(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

###############################################################################
# specimen parameters

const = calculate_energy_constants(300e3)# energy of the beam
const['image_size'] = np.array([2048,2048])
const['aperature'] = np.array([2/3]*2)


atoms,const['lattice'] = read_xyz_file('XYZ files/Au_docahedron.xyz')
const['lattice'] = np.diag(const['lattice']).astype(np.float64)
if np.any(const['lattice']) == None:
    const['lattice'] = np.ceil(np.max(atoms[:,:-1],0))

const['angular'] =  1 /( const['lattice'][:2]) *const['wavel'] * 1e3
const['sampling'] = const['lattice'][:2]/const['image_size']

const['trans_thickness'] = 1 # slice thickness of transmission function
const['no_slices'] = int(np.ceil(const['lattice'][2]/const['trans_thickness']))
const['prop_thickness'] = const['lattice'][2] / const['no_slices'] #  propation distance

const['offset'] = np.array([[0],[0]])
const['extent'] = np.array([[0,(const['image_size'][0]-1)*const['sampling'][0]],
                            [0,(const['image_size'][1]-1)*const['sampling'][1]]])
const['extent'] += const['offset']

print('No Slices: {}'.format(const['no_slices']))

###############################################################################


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


# CTF abberations
ctf_para = {'defocus':-160, 'Cs': -7e-6 * 1e10}
ctf_para['semiangle'] = 40
ctf_para['focal'], ctf_para['angular'] ,ctf_para['gaussian'] = 40, 0, 0

ctf_para = {'defocus':-350, 'Cs': 1.2e-4 * 1e10}
ctf_para['semiangle'] = 10
ctf_para['focal'], ctf_para['angular'] ,ctf_para['gaussian'] = 0, 0, 0


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

# add abberations
conv_exit = CTF_image(conv_exit,ctf_para, const['wavel'],
                      const['sampling'],ctf_para['semiangle'],2,
                      ctf_para['focal'],ctf_para['angular'],ctf_para['gaussian'])
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
                       ctf_para['semiangle'],2,ctf_para['focal'],
                       ctf_para['angular'], ctf_para['gaussian'])

tiles_int = normalise(np.abs(tiles_exit[0])**2)
tiles_ED = diffraction(tiles_exit,const['image_size'],const['lattice'][:2],
                        const['wavel'],const['aperature'],diff_para['mode'])[0]


###############################################################################
# AbTEM multislice

ab_atoms = read('XYZ files/Au_docahedron.xyz')
ab_wave = PlaneWave(energy = const['energy'],gpts=const['image_size'],device='cpu')
ab_pot = Potential(ab_atoms,gpts= const['image_size'],slice_thickness = const['trans_thickness'],
parametrization='kirkland',projection='finite',device='cpu',)
ab_exit_wave = ab_wave.multislice(ab_pot)

ab_ctf = CTF(energy =const['energy'],semiangle_cutoff = ctf_para['semiangle'],
          focal_spread = ctf_para['focal'],defocus = ctf_para['defocus'],
          Cs = ctf_para['Cs'], )
ab_exit_wave = ab_exit_wave.apply_ctf(ab_ctf)

ab_diff = ab_exit_wave.diffraction_pattern(max_angle = diff_para['mode'])

ab_exit_wave = normalise(ab_exit_wave.intensity().array)
ab_diff = ab_diff.array


###############################################################################
# Plotting

x_coords = np.arange(const['image_size'][0]/4,const['image_size'][0]*3/4,).astype(int)
y_coords = np.linspace(const['image_size'][1]/2,const['image_size'][1]/2, int(const['image_size'][1]/2)).astype(int)
diff_snip = np.arange(-diff_para['down_buffer'][1]*2,diff_para['down_buffer'][1]*2,1)


fig = plt.figure(figsize=(15,5))
fig.add_subplot(2,5, 1)
plt.imshow(ab_exit_wave, extent= const['extent'].flatten(),
           cmap='gray',origin='lower',
           vmin=np.min(conv_int),vmax=np.max(conv_int),
           interpolation='nearest')
plt.plot(x_coords*const['sampling'][0], y_coords*const['sampling'][1],'g-')
plt.colorbar(shrink=0.7)
plt.title('(a) abTEM BF',loc='left',fontsize=10)
plt.xlabel('x (\u212b)',fontsize=10)
plt.ylabel('y (\u212b)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


fig.add_subplot(2,5, 2)
plt.imshow(conv_int, extent= const['extent'].flatten(),
           cmap='gray',origin='lower',
           vmin=np.min(conv_int),vmax=np.max(conv_int),
           interpolation='nearest')
plt.plot(x_coords*const['sampling'][0], y_coords*const['sampling'][1],'b--')
plt.colorbar(shrink=0.7)
plt.title('(b) Conv BF',loc='left',fontsize=10)
plt.xlabel('x (\u212b)',fontsize=10)
plt.ylabel('y (\u212b)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


fig.add_subplot(2,5, 3)
plt.imshow(tiles_int, extent= const['extent'].flatten(),
           cmap='gray',origin='lower',
           vmin=np.min(conv_int),vmax=np.max(conv_int),
           interpolation='nearest')
plt.plot(x_coords*const['sampling'][0], y_coords*const['sampling'][1],'r-.')
plt.colorbar(shrink=0.7)
plt.title('(c) Tiling BF',loc='left',fontsize=10)
plt.xlabel('x (\u212b)',fontsize=10)
plt.ylabel('y (\u212b)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


fig.add_subplot(2,5,4)
plt.plot(x_coords*const['sampling'][0],ab_exit_wave[y_coords, x_coords],'g-')
plt.plot(x_coords*const['sampling'][0],conv_int[y_coords, x_coords],'b--')
plt.plot(x_coords*const['sampling'][0],tiles_int[y_coords,x_coords],'r-.')

plt.title('(d) BF (Comparison)',loc='left',fontsize=10)
plt.xlabel('x (\u212b)',fontsize=10)
plt.ylabel('Intensity',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


fig.add_subplot(2,5,5)
plt.plot(x_coords*const['sampling'][0],(conv_int - tiles_int)[y_coords,x_coords],'g-')
plt.plot(x_coords*const['sampling'][0],(ab_exit_wave - tiles_int)[y_coords,x_coords],'b--')

plt.title('(e) BF (Difference)',loc='left',fontsize=10)
plt.xlabel('x (\u212b)',fontsize=10)
plt.ylabel('Intensity',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


fig.add_subplot(2,5,6)
plt.imshow(np.log(ab_diff),extent=diff_para['down_extent'].flatten(),
           cmap='gray',origin='lower',
           interpolation='nearest')
plt.plot(diff_snip*diff_para['sampling'][1],[0]*diff_snip.shape[0],'g-')
plt.title('(f) abTEM ED (log)',loc='left',fontsize=10)
plt.xlabel('\u03B1_x (mrad)',fontsize=10)
plt.ylabel('\u03B1_y (mrad)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(-ctf_para['semiangle'],ctf_para['semiangle'])
plt.ylim(-ctf_para['semiangle'],ctf_para['semiangle'])


fig.add_subplot(2,5,7)
plt.imshow(np.log(conv_ED),extent=diff_para['down_extent'].flatten(),
           cmap='gray',origin='lower',
           interpolation='nearest')
plt.plot(diff_snip*diff_para['sampling'][1],[0]*diff_snip.shape[0],'b--')
plt.title('(g) Conv ED (log)',loc='left')
plt.xlabel('\u03B1_x (mrad)')
plt.ylabel('\u03B1_y (mrad)')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(-ctf_para['semiangle'],ctf_para['semiangle'])
plt.ylim(-ctf_para['semiangle'],ctf_para['semiangle'])


fig.add_subplot(2,5,8)
plt.imshow(np.log(tiles_ED),extent=diff_para['down_extent'].flatten(),
           cmap='gray',origin='lower',
           interpolation='nearest')
plt.plot(diff_snip*diff_para['sampling'][1],[0]*diff_snip.shape[0],'r-.')
plt.title('(h) Tiling ED (log)',loc='left',fontsize=10)
plt.xlabel('\u03B1_x (mrad)',fontsize=10)
plt.ylabel('\u03B1_y (mrad)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(-ctf_para['semiangle'],ctf_para['semiangle'])
plt.ylim(-ctf_para['semiangle'],ctf_para['semiangle'])


fig.add_subplot(2,5,9)
plt.plot(diff_snip*diff_para['sampling'][1],np.log(ab_diff)[int(diff_para['down_size'][0]//2),
                         int(diff_para['down_size'][1]/2-diff_para['down_buffer'][1]*2):
                         int(diff_para['down_size'][1]/2+diff_para['down_buffer'][1]*2),
                                   ],'g-')
    
    

plt.plot(diff_snip*diff_para['sampling'][1],np.log(conv_ED)[int(diff_para['down_size'][0]//2),
                         int(diff_para['down_size'][1]/2-diff_para['down_buffer'][1]*2):
                         int(diff_para['down_size'][1]/2+diff_para['down_buffer'][1]*2),
                                   ],'b--')
    
plt.plot(diff_snip*diff_para['sampling'][1],np.log(tiles_ED)[int(diff_para['down_size'][0]//2),
                         int(diff_para['down_size'][1]/2-diff_para['down_buffer'][1]*2):
                         int(diff_para['down_size'][1]/2+diff_para['down_buffer'][1]*2),
                                    ],'r-.')


plt.title('(i) ED (log Comparison)',loc='left',fontsize=10)
plt.xlabel('\u03B1_x (mrad)',fontsize=10)
plt.ylabel('Intensity (log)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


fig.add_subplot(2,5,10)

plt.plot(diff_snip*diff_para['sampling'][1],(np.log(conv_ED) - np.log(tiles_ED))[int(diff_para['down_size'][0]//2),
                         int(diff_para['down_size'][1]/2-diff_para['down_buffer'][1]*2):
                         int(diff_para['down_size'][1]/2+diff_para['down_buffer'][1]*2),
                                    ],'g-')

plt.plot(diff_snip*diff_para['sampling'][1],(np.log(ab_diff) - np.log(tiles_ED))[int(diff_para['down_size'][0]//2),
                         int(diff_para['down_size'][1]/2-diff_para['down_buffer'][1]*2):
                         int(diff_para['down_size'][1]/2+diff_para['down_buffer'][1]*2),
                                    ],'b--')

    
plt.title('(j) ED (log Difference)',loc='left',fontsize=10)
plt.xlabel('\u03B1_x (mrad)',fontsize=10)
plt.ylabel('Intensity (log)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


