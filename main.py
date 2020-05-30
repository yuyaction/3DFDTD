#coding: UTF-8
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def calc_Efield(Ex,Ey,Ez,Bx,By,Bz):
    #Maxwell-eq
    Ex[1:nx,1:nx,1:nx]=Ex[1:nx,1:nx,1:nx]+c*c*(Bz[1:nx,1:nx,1:nx]-Bz[1:nx,0:nx-1,1:nx]-By[1:nx,1:nx,1:nx]+By[1:nx,1:nx,0:nx-1])-Jx[1:nx,1:nx,1:nx]
    Ey[1:nx,1:nx,1:nx]=Ey[1:nx,1:nx,1:nx]+c*c*(Bx[1:nx,1:nx,1:nx]-Bx[1:nx,1:nx,0:nx-1]-Bz[1:nx,1:nx,1:nx]+Bz[0:nx-1,1:nx,1:nx])-Jy[1:nx,1:nx,1:nx]
    Ez[1:nx,1:nx,1:nx]=Ez[1:nx,1:nx,1:nx]+c*c*(By[1:nx,1:nx,1:nx]-By[0:nx-1,1:nx,1:nx]-Bx[1:nx,1:nx,1:nx]+Bx[1:nx,0:nx-1,1:nx])-Jz[1:nx,1:nx,1:nx]

    #periodic boundary conditions


    return Ex,Ey,Ez

def calc_Bfield(Ex,Ey,Ez,Bx,By,Bz):
    #Maxwell-eq
    Bx[1:nx,1:nx,1:nx]=Bx[1:nx,1:nx,1:nx]-Ez[1:nx,2:nx+1,1:nx]+Ez[1:nx,1:nx,1:nx]+Ey[1:nx,1:nx,2:nx+1]-Ey[1:nx,1:nx,1:nx]
    By[1:nx,1:nx,1:nx]=By[1:nx,1:nx,1:nx]-Ex[1:nx,1:nx,2:nx+1]+Ex[1:nx,1:nx,1:nx]+Ez[2:nx+1,1:nx,1:nx]-Ez[1:nx,1:nx,1:nx]
    Bz[1:nx,1:nx,1:nx]=Bz[1:nx,1:nx,1:nx]-Ey[2:nx+1,1:nx,1:nx]+Ey[1:nx,1:nx,1:nx]+Ex[1:nx,2:nx+1,1:nx]-Ex[1:nx,1:nx,1:nx]

    #periodic boundary conditions

    return Bx,By,Bz





#initial settings
nx = 100
ny = 100
nz = 100
c = 20
simulation_time = 100

Ex = np.zeros((nx+2,ny+2,nz+2))
Ey = np.zeros((nx+2,ny+2,nz+2))
Ez = np.zeros((nx+2,ny+2,nz+2))
Bx = np.zeros((nx+2,ny+2,nz+2))
By = np.zeros((nx+2,ny+2,nz+2))
Bz = np.zeros((nx+2,ny+2,nz+2))
Jx = np.zeros((nx+2,ny+2,nz+2))
Jy = np.zeros((nx+2,ny+2,nz+2))
Jz = np.zeros((nx+2,ny+2,nz+2))

fig = plt.figure()
ax = Axes3D(fig)

for t in range(simulation_time):
    Ex,Ey,Ez = calc_Efield(Ex,Ey,Ez,Bx,By,Bz)
    Bx,By,Bz = calc_Bfield(Ex,Ey,Ez,Bx,By,Bz)


