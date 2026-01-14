#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import scipy
import numba as nb
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pickle

from ase.io import read,write
from ase.atoms import Atoms

import sys
sys.path.append('backbone')
from xyz_reader import calculate_energy_constants
from potential_creator import potential_slices,filter_atoms
from Caching import Cache_original
from tiling_multislice_func import combine_tiles,propagator,get_mask


def normalise(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


@nb.vectorize([nb.complex64(nb.float32), nb.complex128(nb.float64)])
def complex_exp(x):
    return np.cos(x) + 1.j * np.sin(x)


def fft2_convolve_sci(array,kernel):
    array = scipy.fft.fftn(array,axes=(-1,-2), overwrite_x=True,workers=array.shape[1])
    array *= kernel
    array = scipy.fft.ifftn(array,axes=(-1,-2), overwrite_x=True,workers=array.shape[1])
    return array


def multislice_sci(wavefront,tiles,sampling,wavel,aperature,thickness,kappa,sigma):

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

###############################################################################


const = calculate_energy_constants(200e3)
const['image_size'] = np.array([8192,8192])
const['aperature'] = np.array([2/3]*2)

# Amorphous carbon background
from ase.build import bulk 
carbon = bulk('C',cubic=True)
carbon *= (560,560,14)

const['lattice'] = np.ceil(np.max(carbon.positions,0))

carbon.positions[:] += np.random.randn(len(carbon),3)*0.5
carbon.wrap()


const['lattice'] = np.ceil(np.max(carbon.positions,0))
print(const['lattice'])

carbon.set_cell(const['lattice'])
write('Au_part/carbon_{}.xyz'.format(const['lattice']),carbon)


###############################################################################
# Gold Nanoparticle placement

from ase.cluster import Decahedron
cluster = Decahedron('Au', 25,25,0)
cluster_array = Atoms()


rot_array = np.zeros((0,3))


N = (10,10)
for i in range(1,N[0]):
    for j in range(1,N[1]):
        translated_cluster = cluster.copy()
        rot = np.random.randint(-90,90,3)
        translated_cluster.rotate('xyz',rot)
        rot_array = np.vstack((rot_array,rot))
        
        translated_cluster.translate((int(const['lattice'][0]/N[0]*i+np.random.randint(-30,30)),
                                      int(const['lattice'][1]/N[1]*j+np.random.randint(-30,30)),
                                      const['lattice'][2]+np.max(cluster.positions[:,2])))
        cluster_array += translated_cluster
    print('Row: '+str(i))

np.save('Au_part/rot_array.npy',rot_array)
write('Au_part/Au_particle_array_{}.xyz'.format(const['lattice']),cluster_array)


atoms = carbon + cluster_array        
atoms.center(axis=2,vacuum=2)
write('Au_part/Au_array_on_carbon_{}.xyz'.format(const['lattice']),atoms)


atoms = np.concatenate([atoms.copy().positions,
                        atoms.numbers.reshape((atoms.numbers.shape[0],1))],axis=1)
const['lattice'] = np.ceil(np.max(atoms[:,:-1],0))


np.save('Au_part/Au_array_pos_{}'.format(const['lattice']),atoms)

del cluster,translated_cluster,i,j,carbon,cluster_array,rot,rot_array



###############################################################################


# Simulation constants
const['sampling'] = const['lattice'][:2]/const['image_size']
const['angular'] =  1 /( const['lattice'][:2]) *const['wavel'] * 1e3
const['trans_thickness'] = 1
const['no_slices'] = int(np.ceil(const['lattice'][2]/const['trans_thickness']))
const['prop_thickness'] = const['lattice'][2] / const['no_slices'] 

const['offset'] = np.array([[0],[0]])
const['extent'] = np.array([[0,(const['image_size'][0]-1)*const['sampling'][0]],
                            [0,(const['image_size'][1]-1)*const['sampling'][1]]])
const['extent'] += const['offset']

print('\nNo Slices: {}'.format(const['no_slices']))


# tiling constants
const['no_tiles'] = np.array([8,8,1],np.int32)
const['tile_size'] = (const['image_size']/const['no_tiles'][:2]).astype(np.int32)
const['sub_lattice'] = const['lattice']/const['no_tiles']

const['buffer'] = (const['image_size']/32).astype(np.int32)
const['buf_rad'] = const['buffer']*const['sampling']
const['buf_rad'] = np.append(const['buf_rad'],0)
print('Buffer size: {}\n'.format(const['buffer']))

with open('Au_part/Au_array_const{}.pkl'.format(const['lattice']),'wb') as f:
    pickle.dump(const, f)


###############################################################################
# Determine Potential slices

start = timer()
for idx in range(np.prod(const['no_tiles'])):
    start_inside = timer()
    row = idx // const['no_tiles'][0]
    col = idx % const['no_tiles'][0]
    print('Tile No: '+str(idx))
    cache = Cache_original(4096,nb.typed.Dict.empty(key_type=nb.types.int64,
                                    value_type=nb.types.float64[:]))

    atoms_tile = filter_atoms(atoms, const['sub_lattice'], row, col, const['buf_rad'] )
    atoms_tile[:,0] -=  const['sub_lattice'][0] * row - const['buf_rad'][0]
    atoms_tile[:,1] -=  const['sub_lattice'][1] * col - const['buf_rad'][1]

    tile_pot = np.zeros((const['no_slices'],
                          const['tile_size'][0]+2*const['buffer'][0],
                          const['tile_size'][1]+2*const['buffer'][1],),np.float32)

    potential_slices(atoms_tile,tile_pot,const['trans_thickness'],const['sampling'],
                     const['sub_lattice'] + 2*const['buf_rad'] ,cache)

    np.save('Au_part/Au_C_potentials/pot_tile_{}.npy'.format([idx]),tile_pot)
    end_inside = timer()
    print('Tile Saved,time taken: {}'.format(end_inside - start_inside))
cache.clear()
end = timer()
print('\nTime Taken for Potential Slicing {}\n'.format(end - start))
del atoms,atoms_tile,tile_pot,cache,idx, row,col,start_inside,end_inside


###############################################################################
# Tiling-based multislice

start = timer()
for idx in range(np.prod(const['no_tiles'])):
    print('Tile No: '+str(idx))
    wave_tile = np.ones((1,const['tile_size'][0]+2*const['buffer'][0],
                         const['tile_size'][1]+2*const['buffer'][1],),np.complex64)
    
    tile_pot = np.load('Au_part/Au_C_potentials/pot_tile_{}.npy'.format([idx]))

    wave_tile = multislice_sci(wave_tile,tile_pot[None],const['sampling'],const['wavel'],
                               const['aperature'],const['prop_thickness'],
                               const['kappa'],const['sigma'])
    np.save('Au_part/Au_exit_tiles/Au_exit_tile_{}.npy'.format([idx]),wave_tile)

end = timer()
print('\nTime Taken for Multislice {}'.format(end - start))
del tile_pot,wave_tile


###############################################################################
# Combining

tiles_exit =  np.ones((const['no_tiles'][0]*const['no_tiles'][1],
                       const['tile_size'][0]+2*const['buffer'][0],
                       const['tile_size'][1]+2*const['buffer'][1],),
                       dtype=np.complex64)

for idx in range(np.prod(const['no_tiles'])):
    tiles_exit[idx] = np.load('Au_part/Au_exit_tiles/Au_exit_tile_{}.npy'.format([idx]))


start = timer()
tiles_exit = tiles_exit[:,const['buffer'][0]:-const['buffer'][0],
                          const['buffer'][1]:-const['buffer'][1]]

tiles_exit = tiles_exit.reshape((const['no_tiles'][0],const['no_tiles'][1],
                                 const['tile_size'][0],const['tile_size'][1]))
tiles_exit = combine_tiles(tiles_exit,const['no_tiles'],const['tile_size'])[None]

end = timer()
print('Time Taken for Recombining:{}'.format(end - start))
np.save('Au_part/Au_exit_wave_{}'.format(const['no_tiles'][:1]),tiles_exit)


###############################################################################
# Plotting

tiles_int = normalise(np.abs(tiles_exit[0])**2)


fig = plt.figure(figsize=(8,8))
plt.imshow(tiles_int, extent= const['extent'].flatten(),
           cmap='gray',origin='lower',
           vmin=np.min(tiles_int),vmax=np.max(tiles_int),
           interpolation='nearest')
plt.tight_layout()
plt.savefig('Au_part/Au_array_{}.svg'.format(const['no_tiles'][:2]),pad_inches=0)
plt.show()

