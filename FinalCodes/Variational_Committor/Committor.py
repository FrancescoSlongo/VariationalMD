### Libraries used for the code
# Libraries for calculations
from numpy import *
import numpy as np
# Library for precompiled (and faster) code
import numba
from numba import njit

## First guess of the committor function: q increases linearly along x
# The committor starts from the inner edges of the states
@njit(fastmath=True, cache=True)
def qLin(x,y):
    R=0.6
    if x <= R-1.15:
        return 0
    elif x >= 1.15-R:
        return 1
    else:
        return x/(2.0*(1.15-R))+0.5

#Gradient of the first committor guess
@njit(fastmath=True, cache=True)
def dqLin(x,y):
    R=0.6
    if x < R-1.15 or x > 1.15-R:
        return 0.0, 0.0
    else:
        return 1.0/(2.0*(1.15-R)), 0.0
             
@njit(fastmath=True, cache=True)
def qCircleU(x,y):
    R=0.6
    if x == 0 and y == 0:
        return 0.5
    theta = np.arccos(-x/np.sqrt(x**2 + y**2))
    alpha = np.arccos(np.sqrt(1.0 - (R/1.15)**2))
    if theta <= alpha:
        return 0.0
    elif theta > np.pi - alpha:
        return 1.0
    else:
        if y < 0.0:
            return 0.0
        else:
            return (theta - alpha)/(np.pi - alpha)

@njit(fastmath=True, cache=True)
def dqCircleU(x,y):
    R=0.6
    if x == 0.0 and y == 0.0:
        return 0.0, 0.0
    theta = np.arccos(-x/np.sqrt(x**2 + y**2))
    alpha = np.arccos(np.sqrt(1.0 - (R/1.15)**2))
    if theta <= alpha:
        return 0.0, 0.0
    elif theta > np.pi - alpha:
        return 0.0, 0.0
    else: 
        if y > 0.0:
            return y/(x**2 + y**2)/(np.pi - alpha), -x/(x**2 + y**2)/(np.pi - alpha)
        else:
            return 0.0, 0.0            
                
        
# Third guess function: euclidian distance between the two states
@njit(fastmath=True, cache=True)
def qDist(x,y):
    R=0.6
    if (x + 1.15)**2 + y**2 <= R**2:
        return 0.0
    elif (x - 1.15)**2 + y**2 <= R**2:
        return 1.0
    elif (x - 1.15)**2 + y**2 >= (2.0*1.15-R)**2:
        return 0.0
    else:
        return 1.0-(np.sqrt((x - 1.15)**2 + y**2)-R)/(2.0*(1.15-R))
        
@njit(fastmath=True, cache=True)        
def dqDist(x,y):
    R=0.6
    if (x + 1.15)**2 + y**2 <= R**2:
        return 0.0, 0.0
    elif (x - 1.15)**2 + y**2 <= R**2:
        return 0.0, 0.0
    elif (x - 1.15)**2 + y**2 >= (2.0*1.15+R)**2:
        return 0.0, 0.0
    else:
        return -(x - 1.15)/(2.0*(1.15-R)*np.sqrt((x - 1.15)**2 + y**2)), -y/(2.0*(1.15-R)*np.sqrt((x - 1.15)**2 + y**2))
 
@njit(fastmath=True, cache=True)
def ListGuess(n,x,y):
    if n == 0:
        return qLin(x,y)
    elif n == 1:
        return qCircleU(x,y)
    else:
        return qDist(x,y)
    
@njit(fastmath=True, cache=True)
def ListdGuess(n,x,y):
    if n == 0:
        return dqLin(x,y)
    elif n == 1:
        return dqCircleU(x,y)
    else:
        return dqDist(x,y)