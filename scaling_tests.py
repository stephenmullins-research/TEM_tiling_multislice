# -*- coding: utf-8 -*-
"""
timing tests for figure 8
Sequential mode: set workers=1 in fft2_convolve_sci
Parallel mode : set workers=array.shape[0] in fft2_convolve_sci
"""


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import scipy
from ase.io import read
from timeit import default_timer as timer
import pickle

import sys
sys.path.append('backbone')
from xyz_reader import calculate_energy_constants
from potential_creator import potential_slices,potential_tiles
from Caching import Cache_original
from tiling_multislice_func import combine_tiles,get_mask,propagator,complex_exp


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


@nb.vectorize([nb.complex64(nb.float32), nb.complex128(nb.float64)])
def complex_exponential(x):
    return np.cos(x) + 1.j * np.sin(x)


image_list = np.arange(1024,8193,1024)
const = calculate_energy_constants(300e3)#energy of the beam
const['aperature'] = np.array([2/3]*2)
sampling = np.array([0.1,0.1])
no_slices = 100 



conv_time = np.empty(0)
for im in range(image_list.shape[0]):
    const['image_size'] = np.array([image_list[im]]*2)
    print('\nImage Size: {}'.format(const['image_size']))

    # specimen creation
    atoms = read('XYZ files/Fe3O4_OCD_9002332.cif')   
    const['trans_thickness'] = atoms.cell.lengths()[2]
    const['lattice'] = atoms.cell.lengths()
    size_increase = np.array((const['image_size'][0]*sampling[0]//const['lattice'][0],
                              const['image_size'][1]*sampling[1]//const['lattice'][1],
                              1),np.int32)
    atoms *= size_increase
    const['lattice'] = atoms.cell.lengths()
    
    atoms = np.concatenate([atoms.copy().positions,
                            atoms.numbers.reshape((
                            atoms.numbers.shape[0],1))],axis=1)
     
    print('Specimen Size: {}'.format(const['lattice'][:2]))


    # specimen parameters
    const['sampling'] = const['lattice'][:2]/const['image_size']
    const['angular_sampling'] =  1 /( const['lattice'][:2]) *const['wavel'] * 1e3
    const['no_slices'] = int(np.ceil(const['lattice'][2]/const['trans_thickness']))
    const['prop_thickness'] = const['lattice'][2] / const['no_slices'] 
    
    
    # generate potential slices
    conv_pot = np.zeros((const['no_slices'],const['image_size'][0],
                    const['image_size'][1]),np.float32)
    
    cache = Cache_original(4096,nb.typed.Dict.empty(key_type=nb.types.int64,
                                            value_type=nb.types.float64[:]))
    
    start = timer()
    potential_slices(atoms,conv_pot,const['trans_thickness'],
                     const['sampling'],const['lattice'],cache)        

    conv_pot = np.repeat(conv_pot,no_slices,axis=0)
    end = timer()
    print('Time Taken for Potential Slicing:{}'.format(end - start))
    cache.clear()


    # perform multilsice algorithm
    start = timer()
    conv_exit = multislice_sci(conv_pot[None],const['sampling'],const['wavel'],
                               const['aperature'],const['prop_thickness'],
                               const['kappa'],const['sigma'])[0]
    end = timer()
    conv_time = np.append(conv_time,end - start)
    print('Time Taken for Standard Multislice:{}'.format(conv_time[-1]))
    
    del conv_exit, conv_pot,atoms
    with open('Time_Test/Parallel_Conventional.pkl','wb') as f:
        pickle.dump(conv_time, f)



const['no_tiles'] = np.array([2,2,1],np.int32)
tile_time_2 = np.empty(0)
for im in range(image_list.shape[0]):
    const['image_size'] = np.array([image_list[im]]*2)
    print('\nImage Size: {}'.format(const['image_size']))
  
    # specimen creation
    atoms = read('XYZ files/Fe3O4_OCD_9002332.cif')   
    const['trans_thickness'] = atoms.cell.lengths()[2]
    const['lattice'] = atoms.cell.lengths()
    size_increase = np.array((const['image_size'][0]*sampling[0]//const['lattice'][0],
                              const['image_size'][1]*sampling[1]//const['lattice'][1],
                              1),np.int32)
    atoms *= size_increase
    const['lattice'] = atoms.cell.lengths()
  
    atoms = np.concatenate([atoms.copy().positions,
                            atoms.numbers.reshape((
                            atoms.numbers.shape[0],1))],axis=1)
    
    # specimen parameters
    print('Specimen Size: {}'.format(const['lattice']))
    const['sampling'] = const['lattice'][:2]/const['image_size']
    const['angular_sampling'] =  1 /( const['lattice'][:2]) *const['wavel'] * 1e3
    const['no_slices'] = int(np.ceil(const['lattice'][2]/const['trans_thickness']))
    const['prop_thickness'] = const['lattice'][2] / const['no_slices'] 

    # tiling parameters
    const['tile_size']  = (const['image_size'] /const['no_tiles'][:2]).astype(np.int32)
    const['buffer'] = np.ceil(5.96*np.sqrt(const['wavel']*no_slices*const['prop_thickness']/2)
                              /const['sampling']).astype(np.int32)
    const['sub_lattice'] = const['lattice']/const['no_tiles']
    const['buf_rad'] = const['buffer']*const['sampling']
    const['buf_rad'] = np.append(const['buf_rad'],0)
    
    print('Tile size: {}, Buffer size: {}'.format(const['tile_size'],const['buffer']))
    
    # determine potential tiles
    tiles_pot = np.zeros((const['no_tiles'][0]*const['no_tiles'][1],const['no_slices'],
                          const['tile_size'][0]+2*const['buffer'][0],
                          const['tile_size'][1]+2*const['buffer'][1],),np.float32)
    
    cache = Cache_original(4096,nb.typed.Dict.empty(key_type=nb.types.int64,
                                                    value_type=nb.types.float64[:]))
    
    start = timer()
    potential_tiles(tiles_pot,const['no_tiles'],atoms,const['buf_rad'],
                    const['sub_lattice'],const['trans_thickness'],
                    const['sampling'],cache)
    tiles_pot = np.repeat(tiles_pot,no_slices,axis=1)

    end = timer()
    print('Time Taken for Potential Tiling:{}'.format(end - start))
    cache.clear()
  
    # determine electron wave scattering via tiling-based multislice
    start = timer()
    tiles_exit = multislice_sci(tiles_pot,const['sampling'],const['wavel'],
                                const['aperature'], const['prop_thickness'],
                                const['kappa'],const['sigma'])
    # combine tiles
    tiles_exit = tiles_exit[:,const['buffer'][0]:const['tile_size'][0]+const['buffer'][0],
                              const['buffer'][1]:const['tile_size'][1]+const['buffer'][1]]
    
    tiles_exit = tiles_exit.reshape((const['no_tiles'][0],const['no_tiles'][1],
                                     const['tile_size'][0],const['tile_size'][1]))

    tiles_exit = combine_tiles(tiles_exit,const['no_tiles'],const['tile_size'])[None]
    
    end = timer()
    tile_time_2 = np.append(tile_time_2,end - start)
    print('Time Taken for Tiling Multislice:{}'.format(tile_time_2[-1]))
    
    del tiles_exit, tiles_pot, atoms
    with open('Time_Test/Parallel_2x2.pkl','wb') as f:
        pickle.dump(tile_time_2, f)



const['no_tiles'] = np.array([4,4,1],np.int32)
tile_time_4 = np.empty(0)
for im in range(image_list.shape[0]):
    const['image_size'] = np.array([image_list[im]]*2)
    print('\nImage Size: {}'.format(const['image_size']))
  
    # specimen creation
    atoms = read('XYZ files/Fe3O4_OCD_9002332.cif')   
    const['trans_thickness'] = atoms.cell.lengths()[2]
    const['lattice'] = atoms.cell.lengths()
    size_increase = np.array((const['image_size'][0]*sampling[0]//const['lattice'][0],
                              const['image_size'][1]*sampling[1]//const['lattice'][1],
                              1),np.int32)
    atoms *= size_increase
    const['lattice'] = atoms.cell.lengths()
  
    atoms = np.concatenate([atoms.copy().positions,
                            atoms.numbers.reshape((
                            atoms.numbers.shape[0],1))],axis=1)
    
    
    # const['lattice'] = np.round(np.max(atoms[:,:-1],0))
    print('Specimen Size: {}'.format(const['lattice']))
    const['sampling'] = const['lattice'][:2]/const['image_size']
    const['angular_sampling'] =  1 /( const['lattice'][:2]) *const['wavel'] * 1e3
    const['no_slices'] = int(np.ceil(const['lattice'][2]/const['trans_thickness']))
    const['prop_thickness'] = const['lattice'][2] / const['no_slices'] 

    
    const['tile_size']  = (const['image_size'] /const['no_tiles'][:2]).astype(np.int32)

    const['buffer'] = np.ceil(5.96*np.sqrt(const['wavel']*no_slices*const['prop_thickness']/2)
                              /const['sampling']).astype(np.int32)
    const['sub_lattice'] = const['lattice']/const['no_tiles']
    const['buf_rad'] = const['buffer']*const['sampling']
    const['buf_rad'] = np.append(const['buf_rad'],0)
  
    print('Tile size: {}, Buffer size: {}'.format(const['tile_size'],const['buffer']))
    
    
    tiles_pot = np.zeros((const['no_tiles'][0]*const['no_tiles'][1],const['no_slices'],
                          const['tile_size'][0]+2*const['buffer'][0],
                          const['tile_size'][1]+2*const['buffer'][1],),np.float32)
    
    cache = Cache_original(4096,nb.typed.Dict.empty(key_type=nb.types.int64,
                                                    value_type=nb.types.float64[:]))
  
    start = timer()
    potential_tiles(tiles_pot,const['no_tiles'],atoms,const['buf_rad'],const['sub_lattice'],
                    const['trans_thickness'],const['sampling'],cache)
    tiles_pot = np.repeat(tiles_pot,no_slices,axis=1)

    end = timer()
    print('Time Taken for Potential Tiling:{}'.format(end - start))
    cache.clear()

    start = timer()                
 
    tiles_exit = multislice_sci(tiles_pot,const['sampling'],const['wavel'],const['aperature'], 
                                const['prop_thickness'],const['kappa'],const['sigma'])

    tiles_exit = tiles_exit[:,const['buffer'][0]:-const['buffer'][0],
                              const['buffer'][1]:-const['buffer'][1]]
    
    tiles_exit = tiles_exit.reshape((const['no_tiles'][0],const['no_tiles'][1],
                                     const['tile_size'][0],const['tile_size'][1]))
    tiles_exit = combine_tiles(tiles_exit,const['no_tiles'],const['tile_size'])[None]

    end = timer()
    tile_time_4 = np.append(tile_time_4,end - start)
    print('Time Taken for Tiling Multislice:{}'.format(tile_time_4[-1]))
 
    del tiles_exit, tiles_pot, atoms
    with open('Time_Test/Parallel_4x4.pkl','wb') as f:
        pickle.dump(tile_time_4, f)




#Serial or Parallel Computation
fig = plt.figure(figsize=(12,6))
# plt.plot(image_list,abtem_time/60,'k:',label='abTEM')
plt.plot(image_list,conv_time/60,'bo-',label='Conventional')
# plt.plot(image_list,conv_time/60,'b--')
plt.plot(image_list,tile_time_2/60,'gs--',label='2x2 Tiling')
plt.plot(image_list,tile_time_4/60,'k^:',label='4x4 Tiling')
plt.title('CPU Time Comparison: Parallel Processing through {} Potential Slices'.format(no_slices),fontsize=12)
plt.xlabel('Image Size (pixel)',fontsize=12)
plt.ylabel('Time (min)',fontsize=12)
plt.xticks(image_list[::1],fontsize=12)
plt.yticks(fontsize=12)
plt.legend(prop={'size':12})
plt.tight_layout()
plt.savefig('time_scaling_test_cpu_Parallel_{}_slices'.format(no_slices),pad_inches=0)
plt.show()

