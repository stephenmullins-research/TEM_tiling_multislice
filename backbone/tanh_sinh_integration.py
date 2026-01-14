# -*- coding: utf-8 -*-
"""
Module describing the Tanh-Sinh integration method
"""


from numba import njit,prange
import numpy as np

from kirkland_reader import potential_func
from root_finder import Halleys_method_lambertW,Newtons_method
from Caching import key_magic



@njit('float64(float64,float64[:],float64[:],float64[:])', cache=True, fastmath=True)
def error_estimate(eps, value_est, l_sum, r_sum):
    if len(value_est) < 3 or value_est[0] == value_est[-1]:
        return 1.0 if len(value_est) < 3 else 0.0

    e1 = np.abs(value_est[-1] - value_est[-2])
    e2 = np.abs(value_est[-1] - value_est[-3])
    elog = e1 ** (np.log(e1) / np.log(e2))
    e1_2 = e1 ** 2

    l_max, r_max = np.max(np.abs(l_sum)), np.max(np.abs(r_sum))
    e3 = eps * max(l_max, r_max)

    e4 = max(np.abs(l_sum[-1]), np.abs(r_sum[-1]))

    return max(elog, e1_2, e3, e4)



@njit('float64[:,:](float64[:])', cache=True, fastmath=True)
def tanh_sinh_summation(step_controls):
    num_points = int(2 * step_controls[0] + 1)
    absc_weights = np.zeros((num_points, 2), dtype=np.float64)

    for i, k in enumerate(range(-step_controls[0], step_controls[0] + 1)):
        sinh_k = np.sinh(k * step_controls[1])
        cosh_k = np.cosh(k * step_controls[1])

        numerator = step_controls[1] / 2 * np.pi * cosh_k
        denominator = np.cosh(np.pi / 2 * sinh_k) ** 2
        
        absc_weights[i, 0] = np.tanh(np.pi / 2 * sinh_k)
        absc_weights[i, 1] = numerator / denominator

    return absc_weights


@njit('float64[:,:](float64,float64,float64,float64[:,:],int64,int64,float64,int64)', 
      cache=True,fastmath=True)
def tanh_sinh(r0,thickness,cutoff,ele_para,a=-1,b=1,eps=1e-6,step=20):
    #see tanh_sinh on github main.py for expalaination
    h = Newtons_method(eps**2,1e-7,10)#inital step size guess
    order = 0    
    
    for level in range(step):
        lambertw = Halleys_method_lambertW(-(eps ** 2) / h / 2, -1,1e-12,100).real
        j = int(np.log(-2 / np.pi * lambertw) / h)
        
        if level == 0:
            t = np.array([0],dtype=np.float64)
        else:
            t = h * np.arange(1, j + 1, 2)
        
        sinh_t = np.pi / 2 * np.sinh(t)
        cosh_t = np.pi / 2 * np.cosh(t)
        cosh_sinh_t = np.cosh(sinh_t)
        exp_sinh_t = np.exp(sinh_t)
        
        y0 = (b-a)/2 / exp_sinh_t / cosh_sinh_t
        weights = h*(b-a)/2 * cosh_t / cosh_sinh_t ** 2
        
        l_sum = np.sqrt(r0 ** 2 +((a+y0) * thickness/2 + thickness/2) ** 2)
        potential_func(l_sum,l_sum.copy(),cutoff,ele_para)
        l_sum *= weights
        
        r_sum = np.sqrt(r0 ** 2 +((b-y0) * thickness/2 + thickness/2) ** 2)
        potential_func(r_sum,r_sum.copy(),cutoff,ele_para)
        r_sum *= weights
        
        
        if level == 0:
            value_est = l_sum
        else:
            new_est = value_est[-1] / 2 + np.sum(l_sum) + np.sum(r_sum)
            value_est = np.append(value_est,new_est)
        
        error = error_estimate(eps,value_est,l_sum,r_sum)
        
        if abs(error) < eps:
            break
        
        order +=j+1
        h /=2
    
    step_controls = np.array([order-1,h])
    abscissas_weights =  tanh_sinh_summation(step_controls)

    return abscissas_weights


@njit('float64[:](float64[:,:],float64,float64,float64,float64[:,:],float64[:,:])',
        cache=True,fastmath=True)
def integrator(r, ai, bi, cutoff, ele_para,abscissas_weights):        
    zi = np.sqrt(r**2 + (abscissas_weights[:,0] * (bi - ai)/2 + (ai + bi) /2)**2)
    V = np.zeros(zi.shape).flatten()
    
    potential_func(V,zi.flatten(),cutoff,ele_para)
    
    V = V.reshape(zi.shape)
    values = np.sum(V*abscissas_weights[:,1], axis=1)*(bi - ai)/2
    return values



@njit(fastmath=True)
def cache_integrator(cache_class,r, ai, bi, cutoff, ele_para,absc_weights):
    
    key = np.array([bi,ai,cutoff])
    key = key_magic(key,100)

    key_check = sorted(cache_class._cached.keys())
    if key in key_check:
        # The decorated method has been called once with the given args.
        # The calculation will be retrieved from cache.
        result = cache_class.retrieve(key)

        cache_class._hits += 1
    else:
        # The method will be called and its output will be cached.

        result = integrator(r, ai, bi, cutoff, ele_para,absc_weights)
        cache_class.insert(key,result)
        cache_class._misses += 1

    return result


@njit(fastmath=True)
def integration_cached(integral,derivative,limits,ele_para,cutoff,r,absc_weights,cache):
    for i in prange(integral.shape[0]):
        ai = round(limits[i,0],2)
        bi = round(limits[i,1],2)
    
        split = ai * bi < 0
        ai, bi = abs(ai), abs(bi)
        ai, bi = min(ai, bi), min(max(ai, bi), cutoff)
        
        if split:  # split the integral
            V1 = cache_integrator(cache,r,0.0,ai,cutoff,ele_para,absc_weights)
            V2 = cache_integrator(cache,r,0.0,bi,cutoff,ele_para,absc_weights)
            integral[i,:] = V1 + V2 
        else:
            integral[i,:] = cache_integrator(cache,r,ai,bi,cutoff,ele_para,absc_weights)

        for j in range(len(r) - 1):
            derivative[i, j] = (integral[i, j + 1] - integral[i, j]) / (r[j + 1, 0] - r[j, 0])

