# -*- coding: utf-8 -*-
"""
Module describing the different root finding methods 
"""

import numpy as np
from numba import njit

@njit('float64(float64,float64,int64)', cache=True,fastmath=True,nogil=True)
def Newtons_method(tau, tol, max_steps=10):
    #Determine inital guess of step size of the Tanh sinh quadrature
    x = np.log(2 / np.pi * np.log(np.pi / tau))#intial guess

    for _ in range(max_steps):
        fx = np.pi / 2 * np.exp(x) - x - np.log(x * np.pi / tau)#f(x)
        fx_prime = (np.pi / 2 * np.exp(x) - 1 - 1 / x) #f'(x)
        if abs(fx) < tol:
            break

        x -= fx /fx_prime
        
    return x

    
    
@njit('float64(float64,int32,float64,int32)', cache=True,fastmath=True,nogil=True)
def Halleys_method_lambertW(z,k,tol=1e-12,max_steps=100):
    if k==0:
        w=1
    elif k==-1:
        w = np.log(-z)
    
    f0w = w*np.exp(w)-z

    for step in range(max_steps):
        W_h = np.exp(w)*(w+1) - (w+2)*(w*np.exp(w)-z)/(2*w+2)
        w -= f0w /W_h
        f0w = w*np.exp(w)-z
        if abs(f0w) < tol:
            break

    return w

                
#no signature due to input being a function
@njit(cache=True,fastmath=True,nogil=True)
def Brent_Dekker_2_inputs(f,p, x0, x1, max_iter=1000, tol=1e-20):
 
    fx0 = f(x0,p)
    fx1 = f(x1,p)
    
    if abs(fx0) < abs(fx1):
        x0, x1 = x1, x0
        fx0, fx1 = fx1, fx0
 
    x2, fx2 = x0, fx0
 
    mflag = True
    for steps_taken in range(max_iter):
        fx0 = f(x0,p)
        fx1 = f(x1,p)
        fx2 = f(x2,p)
 
        if fx0 != fx2 and fx1 != fx2:
            L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
            L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
            L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
            new = L0 + L1 + L2
 
        else:
            new = x1 - ( (fx1 * (x1 - x0)) / (fx1 - fx0) )
        
        d=x2
        if ((new < ((3 * x0 + x1) / 4) or new > x1) or
            (mflag == True and (abs(new - x1)) >= (abs(x1 - x2) / 2)) or
            (mflag == False and (abs(new - x1)) >= (abs(x2 - d) / 2)) or
            (mflag == True and (abs(x1 - x2)) < tol) or
            (mflag == False and (abs(x2 - d)) < tol)):
            new = (x0 + x1) / 2
            mflag = True
 
        else:
            mflag = False
 
        fnew = f(new,p)
        x2 = x1
 
        if (fx0 * fnew) < 0:
            x1 = new
        else:
            x0 = new
 
        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0
 
        if abs(x1-x0) < tol:
            break
        
    return x0
