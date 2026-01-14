
# -*- coding: utf-8 -*-
"""
multislice simulations of Silicon Polycrystal
Figure 7
"""


import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.spatial import KDTree, Delaunay


from ase.lattice.cubic import Diamond
from ase import Atoms

import sys
sys.path.append('backbone')

from xyz_reader import calculate_energy_constants
from Caching import Cache_original
from potential_creator import potential_slices,potential_tiles
from tiling_multislice_func import multislice_sci,combine_tiles
from contrast_transfer_func import CTF_image ,scherzer_defocus 
from diffraction_func import diffraction,fourier_offset,downsample 

def mse(imageA, imageB):
	return np.sum((imageA - imageB) ** 2)/(imageA.shape[0] * imageA.shape[1])

def normalise(array):
    return ((array - np.min(array)) / (np.max(array) - np.min(array)))

###############################################################################
# specimen creation

diamond = Diamond(symbol='Si', latticeconstant=5.43)
diamond *= (92,)*3
print(diamond.cell)
print(diamond.positions.shape[0])

seed = np.random.uniform(0, diamond.cell[0,0], size=(200, 3))

X, Y, Z = np.meshgrid(np.linspace(0, diamond.cell[0,0], 50),
                      np.linspace(0, diamond.cell[1,1], 50),
                      np.linspace(0, diamond.cell[2,2], 50))
gridpoints = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

nndist, nnidx = KDTree(seed).query(gridpoints)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
atoms_array = Atoms()
for i in np.unique(nnidx):
    region_points = gridpoints[nnidx == i]

    bulk_unit = diamond.copy()
    # Random rotation
    vector = np.random.randint(-5, 5, size=3)
    if np.allclose(vector,0):
        vector = np.random.randint(-5, 5, size=3)

    bulk_unit.rotate((1,0,0), vector, rotate_cell=False)
    bulk_unit.translate(region_points.mean(axis=0) - bulk_unit.get_center_of_mass())

    # Filter atoms inside the cell    
    delaunay = Delaunay(region_points)
    inside_mask = delaunay.find_simplex(bulk_unit.get_positions()) >= 0
    atoms_array += bulk_unit[inside_mask]
    
    # hull = ConvexHull(region_points)
    # polygon = Poly3DCollection(hull.points[hull.simplices], alpha=0.5, 
    #                           facecolors=np.random.uniform(0,1,3),
    #                           linewidths=0.5,edgecolors='gray')

    # ax.add_collection3d(polygon)
# ax.set_xlim([0,  diamond.cell[0,0]])
# ax.set_ylim([0,  diamond.cell[1,1]])
# ax.set_zlim([0,  diamond.cell[2,2]])
# ax.set_box_aspect([1, 1, 1])
# plt.legend()
# plt.tight_layout()
# plt.show()

atoms_array.cell = diamond.cell
# atoms_array.cell = np.max(atoms_array.positions,axis=0)



del diamond,seed,X, Y, Z,gridpoints,nndist, nnidx,
del bulk_unit,region_points,delaunay,inside_mask,vector

###############################################################################
# specimen parameters

const = calculate_energy_constants(200e3)# energy of the beam
const['image_size'] = np.array([2048,2048])
const['aperature'] = np.array([2/3]*2)
const['lattice'] = atoms_array.cell.lengths()

atoms = np.concatenate([atoms_array.copy().positions,
                        atoms_array.numbers.reshape((atoms_array.numbers.shape[0],1))],axis=1)
print('No atoms:{}'.format(atoms.shape[0]))


const['trans_thickness'] = 1
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

# CTF abberations

ctf_para = {'defocus': 0, 'Cs': -6e-6 * 1e10, 'semiangle': 34, 'focal':0,
            'angular':0, 'gauss':0}
ctf_para['defocus'] = scherzer_defocus(ctf_para['Cs'],const['wavel'])


# Diffraction Parameters
diff_para = {'mode' : 'limit'}
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


conv_ctf = CTF_image(conv_exit,ctf_para, const['wavel'],
                      const['sampling'],ctf_para['semiangle'],2,
                      ctf_para['focal'],ctf_para['angular'],ctf_para['gauss'])

conv_int = normalise(np.abs(conv_ctf[0])**2)

conv_ED = diffraction(conv_ctf,const['image_size'],const['lattice'][:2],
                      const['wavel'],const['aperature'],diff_para['mode'])[0]

del conv_pot,cache,conv_exit,conv_ctf
###############################################################################
# tiling parameters


const['no_tiles'] = np.array([2,2,1],np.int32)
const['tile_size'] = (const['image_size']/const['no_tiles'][:2]).astype(np.int32)


const['buffer'] = (const['image_size']/16).astype(np.int32)
print('Buffer size: {}'.format(const['buffer']))

const['sub_lattice'] = const['lattice']/const['no_tiles']
const['buf_rad'] = const['buffer']*const['sampling']
const['buf_rad'] = np.append(const['buf_rad'],0)

diff_para['down_buffer'] = (const['buffer']/const['image_size'])*diff_para['down_size']
diff_para['down_buffer'] = np.round(diff_para['down_buffer']).astype(np.int32)
diff_para['down_buff_rad']  = const['wavel']*1e3/(diff_para['down_buffer']*diff_para['down_samp'])


###############################################################################


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


tiles_ctf = CTF_image(tiles_exit,ctf_para, const['wavel'],const['sampling'],
                        ctf_para['semiangle'],2
                        ,ctf_para['focal'],ctf_para['angular'],ctf_para['gauss'])

tiles_int = normalise(np.abs(tiles_ctf[0])**2)
tiles_ED = diffraction(tiles_ctf,const['image_size'],const['lattice'][:2],
                        const['wavel'],const['aperature'],diff_para['mode'])[0]

del tiles_pot,cache,tiles_exit,tiles_ctf

###############################################################################
# Plotting


print('MSE: {}'.format(mse(conv_int, tiles_int)))

snippet = np.arange(const['tile_size'][0]-const['buffer'][0]*2,
                    const['tile_size'][0]+const['buffer'][0]*2,1)*const['sampling'][0]
diff_snip = np.linspace(-ctf_para['semiangle'],ctf_para['semiangle'],68)


fig = plt.figure(figsize=(12,6))

fig.add_subplot(2,3, 1)
plt.imshow(np.swapaxes(conv_int, 0, 1), extent= const['extent'].flatten(),
           cmap='gray',origin='lower',
           vmin=np.min(conv_int),vmax=np.max(conv_int),
           interpolation='nearest')
plt.plot(snippet,[const['lattice'][0]/2]*snippet.shape[0],'b--')
plt.colorbar(shrink=0.7)
plt.title('(a) Conv BF',loc='left',fontsize=12)
plt.xlabel('x (\u212b)',fontsize=12)
plt.ylabel('y (\u212b)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


fig.add_subplot(2,3,2)
plt.imshow(np.swapaxes(tiles_int, 0, 1), extent= const['extent'].flatten(),
           cmap='gray',origin='lower',
           vmin=np.min(conv_int),vmax=np.max(conv_int),
           interpolation='nearest')
plt.plot(snippet,[const['lattice'][0]/2]*snippet.shape[0],'r-.')
plt.colorbar(shrink=0.7)
plt.title('(b) Tiling BF',loc='left',fontsize=12)
plt.xlabel('x (\u212b)',fontsize=12)
plt.ylabel('y (\u212b)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


fig.add_subplot(2,3,3)
plt.plot(snippet,conv_int[const['tile_size'][0]-const['buffer'][0]*2:
                          const['tile_size'][0]+const['buffer'][0]*2,1024],'b--')
plt.plot(snippet,tiles_int[const['tile_size'][0]-const['buffer'][0]*2:
                           const['tile_size'][0]+const['buffer'][0]*2,1024],'r-.')
plt.title('(c) BF (comparison)',loc='left',fontsize=12)
plt.xlabel('x (\u212b)',fontsize=12)
plt.ylabel('Intensity',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


fig.add_subplot(2,3,4)
plt.imshow(conv_ED**0.1,extent=diff_para['down_extent'].flatten(),
           cmap='gray',origin='lower',
           interpolation='nearest')
plt.plot([-ctf_para['semiangle'],ctf_para['semiangle']],[0,0],'b--')
plt.xlim([-ctf_para['semiangle'],ctf_para['semiangle']])
plt.ylim([-ctf_para['semiangle'],ctf_para['semiangle']])
plt.title('(d) Conv ED (log)',loc='left')
plt.xlabel('\u03B1_x (mrad)')
plt.ylabel('\u03B1_y (mrad)')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


fig.add_subplot(2,3,5)
plt.imshow(tiles_ED**0.1,extent=diff_para['down_extent'].flatten(),
           cmap='gray',origin='lower',
           interpolation='nearest')
plt.plot([-ctf_para['semiangle'],ctf_para['semiangle']],[0,0],'r-.')
plt.xlim([-ctf_para['semiangle'],ctf_para['semiangle']])
plt.ylim([-ctf_para['semiangle'],ctf_para['semiangle']])
plt.title('(e) Tiling ED (log)',loc='left',fontsize=12)
plt.xlabel('\u03B1_x (mrad)',fontsize=12)
plt.ylabel('\u03B1_y (mrad)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


fig.add_subplot(2,3,6)

plt.plot(diff_snip,(conv_ED**0.1)[int(diff_para['down_size'][1]/2)-ctf_para['semiangle']:
                                  int(diff_para['down_size'][1]/2)+ctf_para['semiangle'],
                                  int(diff_para['down_size'][1]/2)],'b--')
    
plt.plot(diff_snip,(tiles_ED**0.1)[int(diff_para['down_size'][1]/2)-ctf_para['semiangle']:
                                   int(diff_para['down_size'][1]/2)+ctf_para['semiangle'],
                                   int(diff_para['down_size'][1]/2)],'r-.')
plt.title('(g) ED (log comparison)',loc='left',fontsize=12)
plt.xlabel('\u03B1_x (mrad)',fontsize=12)
plt.ylabel('Intensity (log)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.tight_layout()
plt.show()


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



