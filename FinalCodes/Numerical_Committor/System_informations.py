### Libraries used for the code
# Libraries for calculations
from numpy import *
import numpy as np
# Library for precompiled (and faster) code
import numba
from numba import njit

@njit(cache=True, fastmath=True)    
def potential(x,y):
    x0=1.0
    a0=1.0/3.0
    b0=5.0/3.0
    u0=5.0
    w0=1.0/5.0
    U = u0*(np.exp(-(x**2 + y**2)) - 3.0/5.0*np.exp(-(x**2 + (y - b0)**2)) - np.exp(-((x-x0)**2+y**2)) - np.exp(-((x+x0)**2 + y**2))) + w0*(x**4 + (y - a0)**4)
    return U

@njit(cache=True, fastmath=True)    
def force(x,y):
    x0=1.0
    a0=1.0/3.0
    b0=5.0/3.0
    u0=5.0
    w0=1.0/5.0
    x2 = x**2  # Square of x coordinate
    y2 = y**2  # Square of y coordinate
    # Exponentials to speed up code
    e1 = np.exp(-(x2 + y2))
    e2 = np.exp(-(x2+(y- b0)**2))
    e3 = np.exp(-((x - x0)**2 + y2))
    e4 = np.exp(-((x + x0)**2 + y2))
    # Components of the force
    fx = 2*u0*(x*e1 -3.0/5.0*x*e2 - (x-x0)*e3 - (x+x0)*e4)-4.0*w0*x**3
    fy = 2*u0*(y*e1 -3.0/5.0*(y - b0)*e2 - y*e3 - y*e4)-4.0*w0*(y - a0)**3
    return fx, fy

@njit(cache=True, fastmath=True) 
def state(x,y): #State of the point. If in reactant or product = 1, otherwise = 0
    R = 0.6
    state = 0  # Transition state
    tmp1 = x**2 + 1.15**2 + y**2
    tmp2 = 2.0*1.15*x
    if tmp1 - tmp2 < R**2:
        # Product state
        state = 1
    if tmp1 + tmp2 < R**2:
        # Reactant state
        state = -1
    return state    