#coding: UTF-8
import math
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def calc_Efield(Ex,Ey,Ez,Bx,By,Bz):
    #Maxwell-eq
    Ex[1:nx,1:nx,1:nx]=Ex[1:nx,1:nx,1:nx]+dt*(c*c*(Bz[1:nx,1:nx,1:nx]-Bz[1:nx,0:nx-1,1:nx]-By[1:nx,1:nx,1:nx]+By[1:nx,1:nx,0:nx-1])-Jx[1:nx,1:nx,1:nx])
    Ey[1:nx,1:nx,1:nx]=Ey[1:nx,1:nx,1:nx]+dt*(c*c*(Bx[1:nx,1:nx,1:nx]-Bx[1:nx,1:nx,0:nx-1]-Bz[1:nx,1:nx,1:nx]+Bz[0:nx-1,1:nx,1:nx])-Jy[1:nx,1:nx,1:nx])
    Ez[1:nx,1:nx,1:nx]=Ez[1:nx,1:nx,1:nx]+dt*(c*c*(By[1:nx,1:nx,1:nx]-By[0:nx-1,1:nx,1:nx]-Bx[1:nx,1:nx,1:nx]+Bx[1:nx,0:nx-1,1:nx])-Jz[1:nx,1:nx,1:nx])

    #periodic boundary conditions


    return Ex,Ey,Ez

def calc_Bfield(Ex,Ey,Ez,Bx,By,Bz):
    #Maxwell-eq
    Bx[1:nx,1:nx,1:nx]=Bx[1:nx,1:nx,1:nx]+dt*(-Ez[1:nx,2:nx+1,1:nx]+Ez[1:nx,1:nx,1:nx]+Ey[1:nx,1:nx,2:nx+1]-Ey[1:nx,1:nx,1:nx])
    By[1:nx,1:nx,1:nx]=By[1:nx,1:nx,1:nx]+dt*(-Ex[1:nx,1:nx,2:nx+1]+Ex[1:nx,1:nx,1:nx]+Ez[2:nx+1,1:nx,1:nx]-Ez[1:nx,1:nx,1:nx])
    Bz[1:nx,1:nx,1:nx]=Bz[1:nx,1:nx,1:nx]+dt*(-Ey[2:nx+1,1:nx,1:nx]+Ey[1:nx,1:nx,1:nx]+Ex[1:nx,2:nx+1,1:nx]-Ex[1:nx,1:nx,1:nx])

    #periodic boundary conditions

    return Bx,By,Bz


def plot2Dfield(field2D):
    plt.imshow(field2D)
    
#initial settings
nx = 100
ny = 100
nz = 100
half_nx = int(nx/2)
half_ny = int(ny/2)
half_nz = int(nz/2)
c = 20
dt = 0.02
dx = 1
dy = 1
dz = 1
inv_dt = 1/dt
inv_dx = 1/dx 
inv_dy = 1/dy
inv_dz = 1/dz
simulation_time = 100

#memory allocation
Ex = np.zeros((nx+2,ny+2,nz+2))
Ey = np.zeros((nx+2,ny+2,nz+2))
Ez = np.zeros((nx+2,ny+2,nz+2))
Bx = np.zeros((nx+2,ny+2,nz+2))
By = np.zeros((nx+2,ny+2,nz+2))
Bz = np.zeros((nx+2,ny+2,nz+2))
Jx = np.zeros((nx+2,ny+2,nz+2))
Jy = np.zeros((nx+2,ny+2,nz+2))
Jz = np.zeros((nx+2,ny+2,nz+2))


for t in range(simulation_time):
    Ez[half_nx,half_ny,half_nz] = Ez[half_nx,half_ny,half_nz] + np.sin(0.5*t)
    Ex,Ey,Ez = calc_Efield(Ex,Ey,Ez,Bx,By,Bz)
    Bx,By,Bz = calc_Bfield(Ex,Ey,Ez,Bx,By,Bz)

    E = np.sqrt(np.square(Ex)+np.square(Ey)+np.square(Ez))
    E_2D = E[:,:,half_nz]

    plot2Dfield(E_2D)
    
    if t ==1:
        #plot settings
        plt.colorbar () # カラーバーの表示
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.pause(.01)

