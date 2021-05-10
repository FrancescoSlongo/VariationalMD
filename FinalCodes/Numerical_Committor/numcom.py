# COMMITTOR FUNCTION FOR 2D SYSTEM - V 2.0
# AUTHORS Luigi Zanovello - Slongo Francesco 02/05/21


# To understand the algorithm you need to follow the reasoning provided here and in the two following videos on how to implement finite differences method to solve elliptic PDEs:	https://www.youtube.com/watch?v=Gd53_TdkmSU
# you probably also need my notes on how I computed the coefficients starting from the backward FP equation

### Libraries used for the code
# Libraries for calculations
from numpy import *
import numpy as np
import numpy.ma as ma
import math
from mpmath import *
import scipy.integrate
import scipy
import scipy.linalg
import glob
import os.path
import sys
import os
# Libraries for the graphical part 
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
# Library for precompiled (and faster) code
import numba
from numba import njit
# Other useful libraries
import sys
import re
import time
import tempfile

## Load potential, force and state
from System_informations import potential, force, state

def fmt(x, pos):
    # format for colorbars tick labels
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)  
        
def main():

    print("Algorithm has started")
    ## Parameters of the run
    temperature=1.0      # Temperature
    diffusion=1.0        # diffusion coefficient
    tstep=0.01           # time interval for MD simulations
    Nsim=1000            # Number of MD simulations to determine boundary conditions
    stepsim=5000         # maximum number of steps for MD simulations
        
    ## Parameters of the grid
    xlimleft=-3.0
    xlimright=3.0
    nxstep=50
    ylimdown=-3.0
    ylimup=3.0
    nystep=50
        
    ## Name of file if we want to save the committor
    savefile=False
    namefile=""
        
    ## Parameters if we want to save committor plot
    saveplot=False
    nameplot=""
    axisticslabelfontsize=9
    axisticslabelfontsizeinset=7
    axislabelfontsize=11
    axislabelfontsizeinset=9
    legendfontsize=7
    lineswidth=2
    ncontour=25
    
    with open("input","r") as f:
        for line in f:
            line=re.sub("#.*$","",line)
            line=re.sub(" *$","",line)
            words=line.split()
            if len(words)==0:
                continue
            key=words[0]
            if key=="temperature":
                temperature=float(words[1])
            elif key=="diffusion":
                diffusion=float(words[1])
            elif key=="tstep":
                tstep=float(words[1])
            elif key=="Nsim":
                Nsim=int(words[1])
            elif key=="stepsim":
                stepsim=int(words[1])
            elif key=="tstep":
                tstep=float(words[1])
            elif key=="xlimleft":
                xlimleft=float(words[1])
            elif key=="xlimright":
                xlimright=float(words[1])
            elif key=="nxstep":
                nxstep=int(words[1])
            elif key=="ylimup":
                ylimup=float(words[1])
            elif key=="ylimdown":
                ylimdown=float(words[1])
            elif key=="nystep":
                nystep=int(words[1])
            elif key=="namefile":
                savefile=True
                namefile=words[1]
            elif key=="nameplot":
                saveplot=True
                nameplot=words[1]
            elif key=="axisticslabelfontsize":
                axisticslabelfontsize=int(words[1])
            elif key=="axisticslabelfontsizeinset":
                axisticslabelfontsizeinset=int(words[1])
            elif key=="axislabelfontsize":
                axislabelfontsize=int(words[1])
            elif key=="axislabelfontsizeinset":
                axislabelfontsizeinset=int(words[1])
            elif key=="legendfontsize":
                legendfontsize=int(words[1])
            elif key=="lineswidth":
                lineswidth=float(words[1]) 
            elif key=="ncontour":
                ncontour=int(words[1])
            else:
                raise Exception("Unknown keyword: "+key)
            
    print("Parameters loaded correctly")
    
    x = np.linspace(xlimleft,xlimright,nxstep)
    y = np.linspace(ylimdown,ylimup,nystep)
    beta = 1.0/temperature # 1/k_{B}T
    dx = x[1]-x[0] # step of the grid along the x direction
    dy = y[1]-y[0] # step of the grid along the y direction
    I = len(x)-1 # index of the last cell along x
    J = len(y)-1 # index of the last cell along y
    N = I*(J+1)+J # index of the last cell in the 1D vector that contains all the I*(J+1)+J+1 = N+1 cells in the system
    q = np.zeros(shape=(N+1)) # vector that contains the N+1 values of the committor for each grid point
    P = np.zeros(shape=(N+1,N+1)) # matrix with (N+1)*(N+1) elements, that determines the linear system to solve P*q = R
    R = np.zeros(shape=(N+1)) # vector with the N+1 elements of results of the backward F-P equation
    bd = 0 # counter for boundary points on which the MD is finished

    ## Precompute some coefficients
    D1 = diffusion*beta*tstep
    D2 = np.sqrt(2.*diffusion*tstep)

    print("Total number of boundary points = ", 2*(nxstep+nystep)-4)
    now=time.time()
    # construction of P and R
    i = 0
    while i < len(x):
        xp = x[i]
        j = 0
        while j < len(y):
            yp = y[j]
            Fx, Fy = force(xp,yp)
            n = i*(J+1)+j					# for each grid point i,j computes the corresponding index n in the 1D vectors and computes the coefficients A,B,C,D,E
            A = diffusion/(dx**2) + (beta*diffusion*Fx)/(2*dx)
            B = 2.*diffusion/(dx**2) + 2.*diffusion/(dy**2)
            C = diffusion/(dx**2) - (beta*diffusion*Fx)/(2*dx)
            D = diffusion/(dy**2) + (beta*diffusion*Fy)/(2*dy)
            E = diffusion/(dy**2) - (beta*diffusion*Fy)/(2*dy)
            if state(xp,yp) != 0:				# if you are inside the reactant basin or the product basin impose boundary condition of q = 0 in Reactant and q = 1 in Product
                P[n,n] = 1.
                if xp > 0.:
                    R[n] = 1.
            elif ((i == 0) or (i == I)):	# if you are on the two boundaries along x, do MD simulations for each boundary point to find the committor on the edges of the grid
                t = 0
                nr = 0
                nt = 0
                while t < Nsim:				# 1000 trajectories for each point
                    xi = x[i]				# for each trajectory in a given grid point on the edge, set as initial conditions the coordinates of that grid point
                    yi = y[j]
                    k = 0
                    while (state(xi,yi)==0 and k < stepsim):	# keep evolving the system untill you enter in Reactant or Product
                        Fx, Fy = force(xi,yi)
                        xf = xi + Fx*D1 + D2*np.random.normal(loc=0.,scale=1.)	# evolve x and y with the equations of motion for a passive particle performing brownian motion in a potential energy landscape
                        yf = yi + Fy*D1 + D2*np.random.normal(loc=0.,scale=1.)
                        xi = xf
                        yi = yf
                        k += 1
                    if xf < 0.:				# if you enter in Reactant
                        nr += 1
                    elif xf > 0.:			# if you enter in Product
                        nt += 1
                    t += 1
                if t != (nt+nr):			# just a check
                    print("Something wrong occurred during MD simulation!")
                bd += 1
                print(bd)					# print the index of the boundary point in which you finished the MD simulations
                P[n,n] = 1.
                R[n] = nt/(nt+nr)			# value of the committor in that point
            elif ((i != 0) and (i != I) and ( (j == 0) or (j == J) )):		# if you are on the boundaries along y do MD simulation for each grid point on the boundary, except those included in the boundaries of x that are already done
                t = 0
                nr = 0
                nt = 0
                while t < Nsim:
                    xi = x[i]
                    yi = y[j]
                    k = 0
                    while (state(xi,yi)==0 and k < stepsim):
                        Fx, Fy = force(xi,yi)
                        xf = xi + Fx*D1 + D2*np.random.normal(loc=0.,scale=1.)
                        yf = yi + Fy*D1 + D2*np.random.normal(loc=0.,scale=1.)
                        xi = xf
                        yi = yf
                        k += 1
                    if xf < 0.:
                        nr += 1
                    elif xf > 0.:
                        nt += 1
                    t += 1
                if t != (nt+nr):
                    print("Something wrong occurred during MD simulation!")
                bd += 1
                print(bd)
                P[n,n] = 1.
                R[n] = nt/(nt+nr)
            else:							# if you are in every other point of the grid fill the P matrix according to the usual pattern
                P[n,(i-1)*(J+1)+j] = C
                P[n,i*(J+1)+j-1] = E
                P[n,i*(J+1)+j] = -B
                P[n,i*(J+1)+j+1] = D
                P[n,(i+1)*(J+1)+j] = A
            j+=1
        i+=1
    
    print("System initialized")

    q = scipy.linalg.solve(P,R)				# solve linear system to find the committor
    for el in q:
        if el > 1.:							# some errors in the solution of the system are inevitable, so don't mind if you get some of these errors, as long as the values of the committor are not too much larger than 1 or too much smaller than 0
            print("Error! Committor larger than 1!")
        elif el < 0.:
            print("Error! Committor smaller than 0!")
    q = np.reshape(q,(I+1,J+1))				# reshape the committor from 1D vector to 2D grid
        
    print("Simulation time ",time.time()-now)
        
      
    # Save committor on a File
    if savefile==True:
        with open(namefile,"w") as f:
            np.savetxt(f,q,fmt="%10.10f")
        print("Committor saved correctly")
        
    
    # Make the plot    
    if saveplot==True:
        xplot = np.empty(shape=(len(x)+1))		# set x and y coordinates for the edges of the bins in the grid
        yplot = np.empty(shape=(len(y)+1))
        k=0
        for el in x:
            xplot[k] = x[k]-(dx/2.)
            k+=1
        xplot[len(x)] = x[len(x)-1]+(dx/2.)
        k=0
        for el in y:
            yplot[k] = y[k]-(dy/2.)
            k+=1
        yplot[len(y)] = y[len(y)-1]+(dy/2.)
        
        with PdfPages(nameplot) as pdf:
            fmt1 = '%r %%'
            fig = plt.figure(figsize=(4.,2.8),dpi=600)
            plt.rc('text')
            panel = fig.add_axes([0.15, 0.15, 0.72, 0.75]) # dimensions and location of the panel within the figure
            cmap = plt.get_cmap('RdBu') # choose colormap for the committor
            pcm = panel.pcolormesh(xplot, yplot, q.T, cmap=cmap,zorder=0,vmin = 0., vmax = 1.) # plot the committor in the grid
            cbar = plt.colorbar(pcm,format=ticker.FuncFormatter(fmt)) # plot colorbar and select format
            cbar.ax.tick_params(labelsize=axisticslabelfontsize-2) # dimension of the labels of the colorbar
            panel.set_xlabel(r'$x$',fontsize=axislabelfontsize,labelpad=2) # labels and ticklabels along x with their fontsize and location, x limits and same for y below
            for tick in panel.xaxis.get_major_ticks(): tick.label.set_fontsize(axisticslabelfontsize)
            panel.set_xlim(xlimleft-0.1,xlimright+0.1)
            panel.xaxis.set_major_locator(MultipleLocator(1))
            panel.xaxis.set_minor_locator(MultipleLocator(0.2))
            panel.set_ylabel(r'$y$',fontsize=axislabelfontsize,labelpad=2)
            for tick in panel.yaxis.get_major_ticks(): tick.label.set_fontsize(axisticslabelfontsize)
            panel.set_ylim(ylimdown-0.1,ylimup+0.1)
            panel.yaxis.set_major_locator(MultipleLocator(1))
            panel.yaxis.set_minor_locator(MultipleLocator(0.2))
            # Grid for contour lines
            x = np.arange(xlimleft,xlimright, 0.025)
            y = np.arange(ylimdown,ylimup, 0.025)
            X, Y = np.meshgrid(x, y)
            # Potential contour lines
            Z = X*0
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i,j] = potential(X[i,j],Y[i,j])
            CS = panel.contour(X, Y, Z, ncontour, colors='k', linewidths = 0.5)
            plt.clabel(CS, fontsize=4, inline=1)
            # Basins contour lines
            P = X*0
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    P[i,j] = state(X[i,j],Y[i,j])
            levelsb = [-1.0, 0.0 ,1.0]
            contour = panel.contour(X, Y, P,levels=levelsb,colors="white",linewidths = 0.5, linestyles='dashed')
            plt.title(r"$q(x,y)$") # title
            pdf.savefig(fig)
            
        print("Committor plot saved correctly")
       
    
main()