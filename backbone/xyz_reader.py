# -*- coding: utf-8 -*-
"""
Module to read XYZ file format
"""

import numpy as np

def calculate_energy_constants(energy):
    """
    Parameters
    ----------
    energy : float/int
        Energy of the electron Beam .

    Returns
    -------
    constants : dict
        A dictionary of constants used in the multislice algorithm.

    """
    
    units = {'Plank const': 6.62607004e-34,
             'light sp': 299792458.0,
             'ele mass': 9.10938356e-31,
             'ele charge': 1.6021766208e-19,
             'Bohr': 0.5291772105638411,
             'kg': 6.0221408585491615e+26,
             'C': 6.241509125883258e+18,
             'sec': 98226947884640.62,
             'Joule': 6.241509125883258e+18,
             'eps0': 8.85418781762039e-12,
             'Amp': 63541.719052630964,
             'm': 10000000000.0,
             }

    constants = {}
    constants['energy'] = energy
    constants['wavel'] = units['Plank const'] * units['light sp'] / np.sqrt(constants['energy'] 
                        * (2 * units['ele mass'] * units['light sp'] ** 2 / 
                           units['ele charge'] + constants['energy'])) / units['ele charge'] * 1e10

    constants['mass'] = (1 + units['ele charge'] * constants['energy'] / (
                        units['ele mass'] * units['light sp'] ** 2)) * units['ele mass']

    constants['sigma'] = (2 * np.pi * constants['mass'] * units['kg'] * units['ele charge'] 
                          * units['C'] * constants['wavel'] / 
                          (units['Plank const'] * units['sec'] * units['Joule']) ** 2)

    constants['eps0'] = units['eps0'] * units['Amp'] ** 2 * units['sec'] ** 4 / (
                        units['kg'] * units['m'] ** 3)


    constants['kappa'] = (4 * np.pi * constants['eps0'] / (
                          2 * np.pi * units['Bohr'] * units['ele charge'] * units['C']))

    return constants


def atomic_symbol_to_number(symbol):
    """
    Converts an atomic symbol to its corresponding atomic number.

    Parameters:
    - symbol (str): Atomic symbol.

    Returns:
    - atomic_number (int): Atomic number corresponding to the symbol.
    """
    atomic_symbols = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
        'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
        'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
        'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
        'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
        'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
    }

    return atomic_symbols.get(symbol, None)

def read_xyz_file(file_path):
    """
    Reads an XYZ file and returns atomic numbers and coordinates.

    Parameters:
    - file_path (str): Path to the XYZ file.

    Returns:
    - atomic_numbers (np.array): Array of atomic numbers.
    - coordinates (np.array): Array of coordinates corresponding to each atomic number.
    """
    data = []

    try:
        with open(file_path, 'r') as f:
            num_atoms = int(f.readline())
            # f.readline()  # Skip the comment line
            # Check if there's a line for lattice vectors
            lattice_line = f.readline()
            if lattice_line.startswith('Lattice='):
                # Split the string by double quotes to extract the part within quotes
                lattice_vectors = lattice_line.split('"')[1]
                lattice_vectors = [float(value) for value in lattice_vectors.split()]
                lattice_vectors = np.array(lattice_vectors).reshape((3,3))
            else:
                lattice_vectors = np.full((3, 3), None)
            
            for _ in range(num_atoms):
                line = f.readline().split()
                symbol = line[0]
                x, y, z = map(float, line[1:4])
                symbol = atomic_symbol_to_number(symbol)
                data.append([x, y, z, symbol])
            
            f.close()
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None
    
    return np.array(data),lattice_vectors

