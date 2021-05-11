### Libraries used for the code
# Libraries for calculations
from numpy import *
import numpy as np
# Library for precompiled (and faster) code
import numba
from numba import njit

from System_informations2 import potential, state

@njit(fastmath=True, cache=True)
def qLin(x,y):
    if x <= -0.73:
        return 0
    elif x >= 0.73:
        return 1
    else:
        return x/(2.0*0.73)+0.5

@njit(fastmath=True, cache=True)    
def dqLin(x,y):
    if x < -0.73 or x > 0.73:
        return 0.0, 0.0
    else:
        return 1.0/(2.0*0.73), 0.0         

@njit(fastmath=True, cache=True)        
def qDist(x,y):
    d1=0.43
    d2=0.85
    if state(x,y)==-1:
        return 0.0
    elif state(x,y)==1:
        return 1.0
    elif (x - 1.15)**2 + y**2 >= (2.0*1.15-d1)**2:
        return 0.0
    elif (x - 1.15)**2 + y**2 <= d2**2:
        return 1.0
    else:
        return 1.0-(np.sqrt((x - 1.15)**2 + y**2)-d2)/(2*1.15-d1-d2)

@njit(fastmath=True, cache=True)
def dqDist(x,y):
    d1=0.43
    d2=0.85
    if state(x,y)==-1:
        return 0.0, 0.0
    elif state(x,y)==1:
        return 0.0, 0.0
    elif (x - 1.15)**2 + y**2 >= (2.0*1.15-d1)**2:
        return 0.0, 0.0
    elif (x - 1.15)**2 + y**2 <= d2**2:
        return 0.0, 0.0
    else:
        return -(x - 1.15)/((2*1.15-d1-d2)*np.sqrt((x - 1.15)**2 + y**2)), -y/((2*1.15-d1-d2)*np.sqrt((x - 1.15)**2 + y**2))
 
@njit(fastmath=True, cache=True)
def ListGuess(n,x,y):
    if n == 0:
        return qLin(x,y)
    else:
        return qDist(x,y)
    
@njit(fastmath=True, cache=True)
def ListdGuess(n,x,y):
    if n == 0:
        return dqLin(x,y)
    else:
        return dqDist(x,y)