#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Effect length of the buffer region on accuracy and computation time
Figure 3
"""


import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from timeit import default_timer as timer
from skimage.metrics import structural_similarity as ssim

import sys
sys.path.append('backbone')

from xyz_reader import read_xyz_file,calculate_energy_constants
from potential_creator import potential_slices,potential_tiles
from Caching import Cache_original
from tiling_multislice_func import multislice_sci,combine_tiles


def mse(imageA, imageB):
	return np.sum((imageA - imageB) ** 2)/(imageA.shape[0] * imageA.shape[1])

def percentage_difference(array1, array2):    
    percentage_diff = 100*(np.abs(array1-array2)/np.maximum(np.abs(array1),np.abs(array2)))    
    return np.mean(percentage_diff)

def normalise(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

###############################################################################
# specimen parameters

const = calculate_energy_constants(300e3)
const['image_size'] = np.array([2048,2048])
const['aperature'] = np.array([2/3]*2)

# specimen generation
atoms,const['lattice'] = read_xyz_file('XYZ files/10x10x10nm.xyz')
const['lattice'] = np.diag(const['lattice'])
if np.any(const['lattice']) == None:
    const['lattice'] = np.ceil(np.max(atoms[:,:-1],0))


const['trans_thickness'] = 1
const['sampling'] = const['lattice'][:2]/const['image_size']
const['angular'] =  const['wavel'] * 1e3/const['lattice'][:2]
const['no_slices'] = int(np.ceil(const['lattice'][2]/const['trans_thickness']))
const['prop_thickness'] = const['lattice'][2] / const['no_slices'] 
print('Sampling: {}'.format(const['sampling']))

const['offset'] = np.array([[0],[0]])
const['extent'] = np.array([[0,(const['image_size'][0]-1)*const['sampling'][0]],
                            [0,(const['image_size'][1]-1)*const['sampling'][1]]])
const['extent'] += const['offset']
print('No Slices: {}'.format(const['no_slices']))

# tiling parameter
const['no_tiles'] = np.array([2,2,1],np.int32)
const['tile_size'] = (const['image_size']/const['no_tiles'][:2]).astype(np.int32)
const['sub_lattice'] = const['lattice']/const['no_tiles']
print('Tile size: {} \n'.format(const['tile_size']))

# Fresnel numbers troughs 
omega =  np.array([1.8775898, 2.74323842, 3.38692584, 3.9418288,  4.41904533, 
                   4.85186964, 5.24030171, 5.61763571, 5.96167554, 6.28351926, 
                   6.59426491, 6.89391251, 7.17136398, 7.44881546, 7.71516888, 
                   7.97042423, 8.21458153, 8.45873883, 8.69179807, 8.91375925, 
                   9.13572044, 9.35768162, 9.56854474, 9.7683098, 9.97917292])
 
 
r = omega*np.sqrt(const['wavel']*const['no_slices']*const['prop_thickness']/2)
buffer_test = np.int32(np.ceil(r/const['sampling'][0]))
print(buffer_test)
buffer_size = const['image_size'][0]/2+(buffer_test*2)
print(buffer_size)

# factors of each buffer
from sympy.ntheory import factorint
factors = [factorint(int(f)) for f in buffer_size]
print(factors)


###############################################################################
# tiling-based multislice with different buffer regions

time_array = np.empty(0)
tiles_int_array = np.zeros((buffer_test.shape[0],const['image_size'][0],const['image_size'][1]))
for b in range(buffer_test.shape[0]):
    
    const['buffer'] = np.array([buffer_test[b]]*2,np.int32)
    const['buf_rad'] = const['buffer']*const['sampling']
    const['buf_rad'] = np.append(const['buf_rad'],0)
    print('Buffer: {}'.format(const['buffer']))
    print('Tile size with Buffer: {}'.format(const['tile_size']+2*const['buffer']))

    # determine potential tiles
    tiles_pot = np.zeros((np.prod(const['no_tiles'][:2]),const['no_slices'],
                          const['tile_size'][0]+2*const['buffer'][0],
                          const['tile_size'][1]+2*const['buffer'][1],),np.float32)
        
    cache = Cache_original(4096,nb.typed.Dict.empty(key_type=nb.types.int64,
                                                value_type=nb.types.float64[:]))
  
    start = timer()
    potential_tiles(tiles_pot,const['no_tiles'],atoms,const['buf_rad'],
                    const['sub_lattice'],const['trans_thickness'],
                    const['sampling'],cache)
    end = timer()
    print('Time Taken for Potential Tiling:{}'.format(end - start))
    cache.clear()
    
    # perform tiling-based multislice
    start = timer()
    tiles_exit = multislice_sci(tiles_pot,const['sampling'],const['wavel'],
                                const['aperature'], const['prop_thickness'],
                                const['kappa'],const['sigma'])
    end = timer()
    time_array = np.append(time_array,end-start)
    print('Time Taken for Tiling Multislice:{}'.format(time_array[-1]))
    
    # combine
    if const['buffer'][0] != 0:
        tiles_exit = tiles_exit[:,const['buffer'][0]:-const['buffer'][0],
                                  const['buffer'][1]:-const['buffer'][1]]
        
    tiles_exit = tiles_exit.reshape((const['no_tiles'][0],const['no_tiles'][1],
                                       const['tile_size'][0],const['tile_size'][1]))
    tiles_exit = combine_tiles(tiles_exit,const['no_tiles'],const['tile_size'])[None]
    

    tiles_int = np.abs(tiles_exit[0])**2
    tiles_int_array[b] = normalise(tiles_int)

    np.save('buffer_test_file/Fresnel_buffer_{}.npy'.format(b),tiles_int)
    del tiles_exit,tiles_int,tiles_pot,cache


###############################################################################
# conventional for comparison metrics

cache = Cache_original(4096,nb.typed.Dict.empty(key_type=nb.types.int64,
                                        value_type=nb.types.float64[:]))

conv_pot = np.zeros((const['no_slices'],const['image_size'][0],
                     const['image_size'][1]),np.float32)

start = timer()
potential_slices(atoms,conv_pot,const['trans_thickness'],
                 const['sampling'],const['lattice'],cache)
end = timer()
print('\nTime Taken for Potential Slicing:{}'.format(end - start))
cache.clear()

start = timer()
conv_exit = multislice_sci(conv_pot[None],const['sampling'],const['wavel'],
                           const['aperature'],const['prop_thickness'],
                           const['kappa'],const['sigma'])
end = timer()
conv_time = end-start
print('Time Taken for Standard Multislice:{}\n'.format(conv_time))
conv_int = np.abs(conv_exit[0])**2
del conv_pot,conv_exit,cache


###############################################################################
# determine values for comparison metrics

mse_array,ssim_array,per_array  = np.empty(0),np.empty(0),np.empty(0)
for b in range(buffer_test.shape[0]):
    tiles_int = np.load('buffer_test_file/Fresnel_buffer_{}.npy'.format(b))
    mse_array = np.append(mse_array,mse(conv_int,tiles_int))
    ssim_array = np.append(ssim_array,ssim(conv_int,tiles_int,
                                           data_range=conv_int.max() - conv_int.min()))
    per_array = np.append(per_array,percentage_difference(conv_int,tiles_int))



###############################################################################
# Plotting 


# Create the figure
fig = plt.figure(figsize=(9, 10))  # Figure size adjusted for uniform spacing
gs = GridSpec(4, 3, figure=fig, wspace=0.2, hspace=0.4)  # More spacing for labels and ticks

ax = fig.add_subplot(gs[0, 0])  # First row, columns 0, 1, 2
ax.plot(buffer_test,mse_array,'bo--')
ax.set_title('(a) MSE',fontsize=12,pad=0.2)
ax.set_xlabel('Buffer Region Length (pixel)',fontsize=10)
ax.set_ylabel('Mean Square Error',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_box_aspect(1)  # Ensure graphs are square

ax = fig.add_subplot(gs[0, 1])  # First row, columns 0, 1, 2
ax.plot(buffer_test,ssim_array,'bo--')
ax.set_title('(b) SSIM',fontsize=12,pad=0.2)
ax.set_xlabel('Buffer Region Length (pixel)',fontsize=10)
ax.set_ylabel('SSIM',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)

ax.set_box_aspect(1)  # Ensure graphs are square

ax = fig.add_subplot(gs[0, 2])  # First row, columns 0, 1, 2
ax.plot(buffer_test,time_array,'bo--')
ax.set_title('(c) Time',fontsize=12,pad=0.2)
ax.set_xlabel('Buffer Region Length (pixel)',fontsize=10)
ax.set_ylabel('Time (seconds)',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)

ax.set_box_aspect(1)  # Ensure graphs are square

# plt.show()

sym = ['d','e','f','g','h','i','j','k','l']
# Add the 3x3 grid of images (remaining 3 rows)
for i in range(3):
    for j in range(3):
        k = i*3+j
        ax = fig.add_subplot(gs[i + 1, j])  # Rows 1-3, columns 0-2
        im = ax.imshow(tiles_int_array[k*2+1,int(const['image_size'][0]/2-32):int(const['image_size'][0]/2+32),
                                    int(const['image_size'][0]/2-32):int(const['image_size'][0]/2+32)],
              cmap='viridis',extent=((np.array([[-32,32],[-32,32]])+1024)*const['sampling']).flatten())
        
        ax.set_title('({}) Buffer size: {}'.format(sym[k],buffer_test[k*2+1]), 
                                              fontsize=12,pad=0.2)
        ax.set_xlabel('x (\u212b)',fontsize=8)
        ax.set_ylabel('y (\u212b)',fontsize=8)


# Adjust layout
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
plt.show()
