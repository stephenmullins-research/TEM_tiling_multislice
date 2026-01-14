# -*- coding: utf-8 -*-
"""
Module to calculate the electrostatic potential and split slices/tiles
"""

import numpy as np
from numba import njit,prange

from kirkland_reader import kirkland_cutoff,kirkland_para
from tanh_sinh_integration import tanh_sinh,integration_cached
from interpolate import rad_interpol_points,interpolate_radial_functions




@njit('float64[:,:](float64[:,:],float64[:],int32[:])', fastmath=True)
def imul(atoms, cell, reps):
    #inspired by from ase/atoms.py __imul__
    #see line 1166 to 1196 on github page

    # Get the number of atoms
    n_atoms = atoms.shape[0]
    
    # Compute the total number of repeated atoms
    total_reps = np.prod(reps)
    
    # Preallocate memory for the repeated atoms
    repeated_atoms = np.empty((n_atoms * total_reps, atoms.shape[1]))
    
    # Repeat atoms along the first axis
    for i in range(total_reps):
        repeated_atoms[i*n_atoms:(i+1)*n_atoms, :] = atoms
    
    # Reposition the repeated atoms 
    i0 = 0
    for x in range(reps[0]):
        for y in range(reps[1]):
            for z in range(reps[2]):                
                repeated_atoms[i0:i0 + n_atoms,:-1] += np.array([x,y,z])*cell
                i0 += n_atoms
    
    return repeated_atoms


@njit('float64[:,:](float64[:,:], float64[:], float64, int32[:])', fastmath=True)
def padding_atoms(atoms, cell, margin, directions=np.array([0, 1, 2])):
    # Determine the number of repetitions needed for padding
    reps = np.array([1, 1, 1], dtype=np.int32)
    for axis in directions:
        reps[axis] = int(1 + 2 * np.ceil(margin / cell[axis]))
    
    # Pad the atoms and reposition
    atoms = imul(atoms,cell,reps)
    atoms[:,:-1] -= cell * np.array([rep // 2 for rep in reps])
    
    # Filter atoms within cell boundaries and margin
    within_bounds = np.ones(atoms.shape[0])
    for axis in directions:
        within_bounds *= atoms[:, axis] > -margin 
        within_bounds *= atoms[:, axis] < cell[axis] + margin

    atoms = atoms[within_bounds==1]
    
    return atoms




@njit('float64[:,:](float64,float64)',fastmath=True,cache=True)
def integral_eval_points(inner_cutoff,cutoff):
    num = int(np.ceil(cutoff / (inner_cutoff *2) * 10.))
    
    r = np.arange(0,num,1,np.float64)
    r *= (np.log10(cutoff) - np.log10(inner_cutoff))/(num-1)
    r+= np.log10(inner_cutoff)
    r[-1] = np.log10(cutoff)
    r = np.power(10,r)
    r[0],r[-1] = inner_cutoff,cutoff

    return r.reshape((r.shape[0],1))



#no signature due to multi outputs
@njit(fastmath=True)
def atom_split(no_slices, atoms_slice, thickness, cutoff, vol):
    
    pos = np.zeros((0, 3))
    integral_limits = np.zeros((0, 2))
    rle = np.zeros((no_slices+1,),np.int32)

    for k in prange(no_slices):
        # determine if atoms are in /partially in a slice
        in_slice = (atoms_slice[:, 2] >= thickness * k - cutoff)
        in_slice *= (atoms_slice[:, 2] < thickness * (k + 1) + cutoff)
        atoms_layer = atoms_slice.copy()[in_slice]

        # fill vacuum in slice
        direction = np.array([0, 1], np.int32)
        atoms_layer = padding_atoms(atoms_layer, vol, cutoff, direction)


        # Compute enterance and exits values for each atom layer
        limits = np.zeros((atoms_layer.shape[0], 2))
        for i in prange(atoms_layer.shape[0]):
            limits[i, 0] = max(thickness * k, 0) - atoms_layer[i, 2]
            limits[i, 1] = min(thickness * (k + 1), vol[2]) - atoms_layer[i, 2]
     
        pos = np.vstack((pos, atoms_layer[:, :-1]))
        integral_limits = np.vstack((integral_limits, limits))
        rle[k + 1] = rle[k] + atoms_layer.shape[0]

    return pos, integral_limits, rle



@njit(fastmath=True)#'void(float64[:,:],float32[:,:,:],float64,float64[:],float64[:])',
def potential_slices(atoms,slices,thickness,sampling,vol,cache):
    
    elements = np.unique(atoms[:,-1]).astype(np.int32)
    
    max_cut = 0.0
    for number in elements:
        max_cut = max(kirkland_cutoff(number), max_cut)

    direction = np.array([2],np.int32)
    atoms = padding_atoms(atoms,vol,max_cut,direction)
    
    for number in elements:
        #select the atoms of this element
        atoms_slice = atoms.copy()[atoms[:,-1] == number] 
        
        #atomic radius as a cutoff and atomic scattering parameters.
        cutoff = kirkland_cutoff(number) 
        para = kirkland_para(number)
        
        #split the atoms into each slice
        pos,integral_limits,rle = atom_split(slices.shape[0],atoms_slice,
                                             thickness,cutoff,vol)

        #determine evaluation points for integral
        inner_cutoff = min(sampling)/2
        eval_points = integral_eval_points(inner_cutoff, cutoff)
  
        # create arrays for integral and derivatives
        integral = np.zeros((pos.shape[0],eval_points.shape[0]),np.float64)
        derivative = np.zeros((pos.shape[0],eval_points.shape[0]),np.float64)
        
        #get weights and abscissas of  tanh-sinh quadrature
        abscissas_weights = tanh_sinh(min(sampling)/2,thickness,cutoff,para,-1,1,1e-6,20)

        #integration to determine projected potential and gradient
        integration_cached(integral,derivative,integral_limits,
                           para,cutoff,eval_points,abscissas_weights,cache)
          
        # determine radial disc for interpolation
        disc_indices = rad_interpol_points(min(sampling),cutoff)
        # convert from real space to image space with interpolation        
        interpolate_radial_functions(slices,rle,disc_indices, pos,
                                     integral.astype(np.float32),
                                     derivative.astype(np.float32),
                                     eval_points[:,0].astype(np.float32),
                                     sampling.astype(np.float32))



@njit('float64[:,:](float64[:,:],float64[:],int64,int64,float64[:])',
      fastmath=True, parallel=True)
def filter_atoms(atoms, sub_vol, row, col, buffer):
    """
    

    Parameters
    ----------
    atoms : array
        Cartesian coordinates of all atoms in the model.
    sub_vol : Array
        Volume of a tile.
    row : int
        coordinate of the tile in an array along x axis.
    col : int
        coordinate of the tile in an array along y axis.
    buffer : 2-D array
        real size of the buffer region.

    Returns
    -------
    Array
        Returns all atoms in the tile volume.

    """
    check = np.ones(atoms.shape[0])

    left = sub_vol[0] * row - buffer[0]
    right = sub_vol[0] * (row + 1) + buffer[0]
    
    lower = sub_vol[1] * col - buffer[1]
    upper = sub_vol[1] * (col + 1) + buffer[1]

    for a in prange(atoms.shape[0]):                
        check[a] *= left < atoms[a,0] < right
        check[a] *= lower < atoms[a,1] < upper
    
    return atoms[check==1]

#'void(float32[:,:,:,:],int32[:],float64[:,:],float64[:],float64[:],float64,float64[:])')
@njit(fastmath=True)
def potential_tiles(tiles,no_tiles,atoms,buffer,sub_vol,thickness,sampling,cache):
    """
    function that orgainises atoms into the lateral potential tiles. 
    And determines the electrostatic potential 

    Parameters
    ----------
    tiles : 4D float32 Array 
        NumPy array for the electostatic potential tiles.
    no_tiles : 1D int32 Array
        Number of tiles the specimen will be cut into.
    atoms : 2D float64 Array
        Cartesian coordinates and Atomic number of all atoms in the specimen.
    buffer : 1D Array
        Physical Lenght of the buffer region.
    sub_vol : 1D Array
        Volume of a single tile (not including buffer region).
    thickness : float64
        slice thickness.
    sampling : 1D float64 Array
        Angstrom per pixel.
    cache : Class
        Caching class.

    Returns
    -------
    None.

    """
    direction = np.array([0,1],np.int32)
    # Make sure all available space is filled in the specimen
    atoms = padding_atoms(atoms,sub_vol*no_tiles,max(buffer),direction)
        
    for idx in prange(no_tiles[0] * no_tiles[1]):
        row = idx // no_tiles[0]
        col = idx % no_tiles[0]
        
        # determine which atoms are in the tile
        atoms_tile = filter_atoms(atoms, sub_vol, row, col, buffer)
        
        #reposition atoms to within the tile volume.
        atoms_tile[:,0] -= sub_vol[0] * row - buffer[0]
        atoms_tile[:,1] -= sub_vol[1] * col - buffer[1]
        
        # Determine the Potential slices in the tile volume
        potential_slices(atoms_tile,tiles[idx],thickness,sampling,
                         sub_vol + buffer*2,cache)

