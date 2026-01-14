# -*- coding: utf-8 -*-
"""
Module for Caching 
"""

import numpy as np
from numba import int32,float32,float64,types,typed,njit
from numba.experimental import jitclass


spec = [('_max_size',int32),
        ('_key', float64[:,:]),
        ('_values', float32[:,:]),
        ('_hits', int32),
        ('_misses', int32),
        ]

@jitclass(spec)
class Cache:

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._key = np.empty((0,3),np.float64)
        self._values = np.empty((1,0),np.float32)
        self._hits = 0
        self._misses = 0

    @property
    def hits(self):
       #Number of times a previously calculated object was retrieved.
        return self._hits

    @property
    def misses(self):
        #Number of times a new object had to be calculated.
        return self._hits
    
    @property
    def key(self):
        # Array that stores the key
        return self._key
    
    @property
    def values(self):
        #Array that stores the object to cache
        return self._values
    
    def insert(self, key, value):
        """
        key : The dictionary key of the cached object.
        value : The object to cache.
        """
        self._key = np.append(self._key,key,0)
        
        if self._values.shape[1] != value.shape[1]:
            self._values = np.append(self._values,value,1)
        else:
            self._values = np.append(self._values,value,0)
        
        #self._check_size()

    def retrieve(self, idx):
        """
        Parameters -> The index key of the cached item
        Returns -> The cached object.
        """

        value = self._values[idx,:]
        return value

    def _check_size(self):
        #Delete 1st item from cache, if it is too large.
        if self._max_size is not None:
            while self._key.shape[0] > self._max_size:
                self._key = self._key[1:,:]
                self._values = self._values[1:,:]

    def clear(self):
       # Clear the cache
        self._key = np.empty((0,3),np.float64)
        self._values = np.empty((1,0),np.float32)
        self._hits = 0
        self._misses = 0





spec_oro = [('_max_size',int32),
            ('_cached',types.DictType(*(types.float64,
                                        types.float64[:]))),
            ('_hits', int32),
            ('_misses', int32),
            ]

@jitclass(spec_oro)
class Cache_original:
    """
    Cache object.
    Class for handling a dictionary-based cache. When the cache is full, the first inserted item is deleted.
    Parameters
    ----------
    max_size : int
        The maximum number of values stored by this cache.
    """

    def __init__(self, max_size: int,dictionary):
        self._max_size = max_size
        self._cached = dictionary
        self._hits = 0
        self._misses = 0

    @property
    def cached(self) -> dict:
        """
        Dictionary of cached data.
        """
        return self._cached

    @property
    def hits(self) -> int:
        """
        Number of times a previously calculated object was retrieved.
        """
        return self._hits

    @property
    def misses(self) -> int:
        """
        Number of times a new object had to be calculated.
        """
        return self._hits

    def __len__(self) -> int:
        """
        Number of objects cached.
        """
        return len(self._cached)

    def insert(self, key, value):
        """
        Insert new value into the cache.
        Parameters
        ----------
        key : Any
            The dictionary key of the cached object.
        value : Any
            The object to cache.
        """
        self._cached[key] = value
        
        self._check_size()

    def retrieve(self, key):
        """
        Retrieve object from cache.
        Parameters
        ----------
        key: Any
            The key of the cached item.
        Returns
        -------
        Any
            The cached object.
        """
        return self._cached[key]

    def _check_size(self):
        """
        Delete item from cache, if it is too large.
        """
        if self._max_size is not None:
            while len(self._cached) > self._max_size:
                self._cached.popitem()

    def clear(self):
        """
        Clear the cache.
        """
        self._cached = typed.Dict.empty(*(types.float64,types.float64))
        self._hits = 0
        self._misses = 0

@njit
def key_magic(aList, base=100):
    n= 0
    for d in range(aList.shape[0]):
        n = base*n + aList[d]
    return n
