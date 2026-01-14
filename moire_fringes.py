# -*- coding: utf-8 -*-


"""
multislice simulations of Iron crystals 
to show simulated interference fringes
Figure 5

"""


from ase import Atoms
from ase.build import bulk
from ase.visualize import view

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import sys
sys.path.append('backbone')

from xyz_reader import calculate_energy_constants
from Caching import Cache_original
from potential_creator import potential_slices,potential_tiles
from tiling_multislice_func import multislice_sci,combine_tiles
from contrast_transfer_func import CTF_image
from diffraction_func import diffraction,fourier_offset,downsample 


def normalise(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


###############################################################################
# specimen creation 


# Fe a= 3.66
# Create two bulk diamond iron crystals
Fe1 = bulk('Fe', 'fcc', a=3.66, cubic=True)
Fe1 *= (40,40,40)

# Define a rotation for the second crystal (e.g., 2 degrees around [001])
# angle = -5  # degrees
Fe2 = Fe1.copy()
Fe2.rotate(a=5, v='z', center='COP', rotate_cell=False)

filtered_atoms = []
for atom in Fe2:
    if -1e-12 <= atom.position[0] <= Fe2.cell[0,0]:  # Keep atoms inside bounds
        if -1e-12 <= atom.position[1] <= Fe2.cell[1,1]:  # Keep atoms inside bounds
            if -1e-12 <= atom.position[2] <= Fe2.cell[2,2]:  # Keep atoms inside bounds
                filtered_atoms.append(atom)

Fe2 = Atoms([atom for atom in filtered_atoms], cell=Fe2.cell, pbc=[1, 1, 1])

# Translate the second crystal slightly to avoid full overlap
Fe2.translate([0.0, 0.0, Fe1.cell[2, 2]])


supercell = Fe1 + Fe2
supercell.cell = np.max(supercell.positions,axis=0)
supercell.center(vacuum=3.66/2)
view(supercell)


atoms = np.concatenate([supercell.copy().positions,
                        supercell.numbers.reshape((
                        supercell.numbers.shape[0],1))],axis=1)


###############################################################################

# specimen parameters

const = calculate_energy_constants(200e3)# energy of the beam
const['image_size'] = np.array([2048,2048])
const['aperature'] = np.array([2/3]*2)

const['lattice'] = supercell.cell.lengths()
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

# CTF abberations
ctf_para = {'defocus': 300, 'Cs':  -5e-4 * 1e10, 'semiangle': 40, 'focal':5,
            'angular':0.1, 'gauss':2}

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

# add abberations
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
                       ctf_para['semiangle'],2,ctf_para['focal'],
                       ctf_para['angular'], ctf_para['gaussian'])

tiles_int = normalise(np.abs(tiles_exit[0])**2)
tiles_ED = diffraction(tiles_exit,const['image_size'],const['lattice'][:2],
                        const['wavel'],const['aperature'],diff_para['mode'])[0]


###############################################################################
# Plotting


x_coords = np.arange(const['image_size'][0]/4,const['image_size'][0]*3/4,).astype(int)
y_coords = np.linspace(const['image_size'][1]/2,const['image_size'][1]/2, int(const['image_size'][1]/2)).astype(int)
diff_snip = np.arange(-diff_para['down_buffer'][1]*2,diff_para['down_buffer'][1]*2,1)


fig = plt.figure(figsize=(12,6))
fig.add_subplot(2,4, 1)
plt.imshow(conv_int, extent= const['extent'].flatten(),
           cmap='gray',origin='lower',
           vmin=np.min(conv_int),vmax=np.max(conv_int),
           interpolation='nearest')
plt.plot(x_coords*const['sampling'][0], y_coords*const['sampling'][1],'b--')
plt.colorbar(shrink=0.7)
plt.title('(a) Conv BF',loc='left',fontsize=12)
plt.xlabel('x (\u212b)',fontsize=12)
plt.ylabel('y (\u212b)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


fig.add_subplot(2,4,2)
plt.imshow(tiles_int, extent= const['extent'].flatten(),
           cmap='gray',origin='lower',
           vmin=np.min(conv_int),vmax=np.max(conv_int),
           interpolation='nearest')
plt.plot(x_coords*const['sampling'][0], y_coords*const['sampling'][1],'r-.')
plt.colorbar(shrink=0.7)
plt.title('(b) Tiling BF',loc='left',fontsize=12)
plt.xlabel('x (\u212b)',fontsize=12)
plt.ylabel('y (\u212b)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


fig.add_subplot(2,4,3)
plt.plot(x_coords*const['sampling'][0],conv_int[y_coords, x_coords],'b--')
plt.plot(x_coords*const['sampling'][0],tiles_int[y_coords,x_coords],'r-.')
plt.title('(c) BF (comparison)',loc='left',fontsize=12)
plt.xlabel('x (\u212b)',fontsize=12)
plt.ylabel('Intensity',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


fig.add_subplot(2,4,4)
plt.plot(x_coords*const['sampling'][0],(conv_int-tiles_int)[y_coords, x_coords],'g-')
plt.title('(d) BF (Difference)',loc='left',fontsize=12)
plt.xlabel('x (\u212b)',fontsize=12)
plt.ylabel('Intensity',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ticklabel_format(style="plain",  axis="y")


fig.add_subplot(2,4,5)
plt.imshow(np.log(conv_ED),extent=diff_para['down_extent'].flatten(),
           cmap='gray',origin='lower',
           interpolation='nearest')
plt.plot(diff_snip*diff_para['sampling'][1],[0]*diff_snip.shape[0],'b--')
plt.title('(e) Conv ED (log)',loc='left')
plt.xlabel('\u03B1_x (mrad)')
plt.ylabel('\u03B1_y (mrad)')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-ctf_para['semiangle'],ctf_para['semiangle'])
plt.ylim(-ctf_para['semiangle'],ctf_para['semiangle'])


fig.add_subplot(2,4,6)
plt.imshow(np.log(tiles_ED),extent=diff_para['down_extent'].flatten(),
           cmap='gray',origin='lower',
           interpolation='nearest')
plt.plot(diff_snip*diff_para['sampling'][1],[0]*diff_snip.shape[0],'r-.')
plt.title('(f) Tiling ED (log)',loc='left',fontsize=12)
plt.xlabel('\u03B1_x (mrad)',fontsize=12)
plt.ylabel('\u03B1_y (mrad)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-ctf_para['semiangle'],ctf_para['semiangle'])
plt.ylim(-ctf_para['semiangle'],ctf_para['semiangle'])


fig.add_subplot(2,4,7)
plt.plot(diff_snip*diff_para['sampling'][1],np.log(conv_ED)[int(diff_para['down_size'][0]//2),
                         int(diff_para['down_size'][1]/2-diff_para['down_buffer'][1]*2):
                         int(diff_para['down_size'][1]/2+diff_para['down_buffer'][1]*2),
                                   ],'b--')
    
plt.plot(diff_snip*diff_para['sampling'][1],np.log(tiles_ED)[int(diff_para['down_size'][0]//2),
                         int(diff_para['down_size'][1]/2-diff_para['down_buffer'][1]*2):
                         int(diff_para['down_size'][1]/2+diff_para['down_buffer'][1]*2),
                                    ],'r-.')
    
    
plt.title('(g) ED (log comparison)',loc='left',fontsize=12)
plt.xlabel('\u03B1_x (mrad)',fontsize=12)
plt.ylabel('Intensity (log)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fig.add_subplot(2,4,8)
plt.plot(diff_snip*diff_para['sampling'][1],(np.log(conv_ED)-np.log(tiles_ED))[int(diff_para['down_size'][0]//2),
                         int(diff_para['down_size'][1]/2-diff_para['down_buffer'][1]*2):
                         int(diff_para['down_size'][1]/2+diff_para['down_buffer'][1]*2),
                                   ],'g-')
    

plt.title('(h) ED (log Difference)',loc='left',fontsize=12)
plt.xlabel('\u03B1_x (mrad)',fontsize=12)
plt.ylabel('Intensity (log)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.tight_layout()
plt.show()

