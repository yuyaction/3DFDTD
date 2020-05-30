#coding: UTF-8
import math
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def calc_Efield(Ex,Ey,Ez,Bx,By,Bz):
    #Maxwell-eq
    Ex[1:nx,1:ny,1:nz]=Ex[1:nx,1:ny,1:nz]+dt*(c*c*(inv_dz*(Bz[1:nx,1:ny,1:nz]-Bz[1:nx,0:ny-1,1:nz])+inv_dy*(-By[1:nx,1:ny,1:nz]+By[1:nx,1:ny,0:nz-1]))-Jx[1:nx,1:ny,1:nz])
    Ey[1:nx,1:ny,1:nz]=Ey[1:nx,1:ny,1:nz]+dt*(c*c*(inv_dx*(Bx[1:nx,1:ny,1:nz]-Bx[1:nx,1:ny,0:nz-1])+inv_dz*(-Bz[1:nx,1:ny,1:nz]+Bz[0:nx-1,1:ny,1:nz]))-Jy[1:nx,1:ny,1:nz])
    Ez[1:nx,1:ny,1:nz]=Ez[1:nx,1:ny,1:nz]+dt*(c*c*(inv_dy*(By[1:nx,1:ny,1:nz]-By[0:nx-1,1:ny,1:nz])+inv_dx*(-Bx[1:nx,1:ny,1:nz]+Bx[1:nx,0:ny-1,1:nz]))-Jz[1:nx,1:ny,1:nz])

    #periodic boundary conditions
    Ex[1:nx,1:ny,nz+1] = Ex[1:nx,1:ny,1   ]
    Ex[1:nx,1:ny,0   ] = Ex[1:nx,1:ny,nz  ]
    Ex[1:nx,ny+1,1:nz] = Ex[1:nx,1   ,1:nz]
    Ex[1:nx,0   ,1:nz] = Ex[1:nx,ny  ,1:nz]
    Ex[nx+1,1:ny,1:nz] = Ex[1   ,1:ny,1:nz]
    Ex[0   ,1:ny,1:nz] = Ex[nx  ,1:ny,1:nz]

    Ey[1:nx,1:ny,nz+1] = Ey[1:nx,1:ny,1   ]
    Ey[1:nx,1:ny,0   ] = Ey[1:nx,1:ny,nz  ]
    Ey[1:nx,ny+1,1:nz] = Ey[1:nx,1   ,1:nz]
    Ey[1:nx,0   ,1:nz] = Ey[1:nx,ny  ,1:nz]
    Ey[nx+1,1:ny,1:nz] = Ey[1   ,1:ny,1:nz]
    Ey[0   ,1:ny,1:nz] = Ey[nx  ,1:ny,1:nz]
    
    Ez[1:nx,1:ny,nz+1] = Ez[1:nx,1:ny,1   ]
    Ez[1:nx,1:ny,0   ] = Ez[1:nx,1:ny,nz  ]
    Ez[1:nx,ny+1,1:nz] = Ez[1:nx,1   ,1:nz]
    Ez[1:nx,0   ,1:nz] = Ez[1:nx,ny  ,1:nz]
    Ez[nx+1,1:ny,1:nz] = Ez[1   ,1:ny,1:nz]
    Ez[0   ,1:ny,1:nz] = Ez[nx  ,1:ny,1:nz]
    return Ex,Ey,Ez

def calc_Bfield(Ex,Ey,Ez,Bx,By,Bz):
    #Maxwell-eq
    Bx[1:nx,1:ny,1:nz]=Bx[1:nx,1:ny,1:nz]+dt*(inv_dz*(-Ez[1:nx,2:ny+1,1:nz]+Ez[1:nx,1:ny,1:nz])+inv_dy*(Ey[1:nx,1:ny,2:nz+1]-Ey[1:nx,1:ny,1:nz]))
    By[1:nx,1:ny,1:nz]=By[1:nx,1:ny,1:nz]+dt*(inv_dx*(-Ex[1:nx,1:ny,2:nz+1]+Ex[1:nx,1:ny,1:nz])+inv_dz*(Ez[2:nx+1,1:ny,1:nz]-Ez[1:nx,1:ny,1:nz]))
    Bz[1:nx,1:ny,1:nz]=Bz[1:nx,1:ny,1:nz]+dt*(inv_dy*(-Ey[2:nx+1,1:ny,1:nz]+Ey[1:nx,1:ny,1:nz])+inv_dx*(Ex[1:nx,2:ny+1,1:nz]-Ex[1:nx,1:ny,1:nz]))

    #periodic boundary conditions
    Bx[1:nx,1:ny,nz+1] = Bx[1:nx,1:ny,1   ]
    Bx[1:nx,1:ny,0   ] = Bx[1:nx,1:ny,nz  ]
    Bx[1:nx,ny+1,1:nz] = Bx[1:nx,1   ,1:nz]
    Bx[1:nx,0   ,1:nz] = Bx[1:nx,ny  ,1:nz]
    Bx[nx+1,1:ny,1:nz] = Bx[1   ,1:ny,1:nz]
    Bx[0   ,1:ny,1:nz] = Bx[nx  ,1:ny,1:nz]

    By[1:nx,1:ny,nz+1] = By[1:nx,1:ny,1   ]
    By[1:nx,1:ny,0   ] = By[1:nx,1:ny,nz  ]
    By[1:nx,ny+1,1:nz] = By[1:nx,1   ,1:nz]
    By[1:nx,0   ,1:nz] = By[1:nx,ny  ,1:nz]
    By[nx+1,1:ny,1:nz] = By[1   ,1:ny,1:nz]
    By[0   ,1:ny,1:nz] = By[nx  ,1:ny,1:nz]
    
    Bz[1:nx,1:ny,nz+1] = Bz[1:nx,1:ny,1   ]
    Bz[1:nx,1:ny,0   ] = Bz[1:nx,1:ny,nz  ]
    Bz[1:nx,ny+1,1:nz] = Bz[1:nx,1   ,1:nz]
    Bz[1:nx,0   ,1:nz] = Bz[1:nx,ny  ,1:nz]
    Bz[nx+1,1:ny,1:nz] = Bz[1   ,1:ny,1:nz]
    Bz[0   ,1:ny,1:nz] = Bz[nx  ,1:ny,1:nz]

    return Bx,By,Bz


def plot2Dfield(field2D,t):
    plt.imshow(field2D)
    time_now = "Time: " + str(round(t*dt,2))
    plt.title(time_now)
    
def plot3Dvector(fieldX,fieldY,fieldZ,t):
    for i in range(0,nx+1,10):
        for j in range(0,ny+1,10):
            for k in range(0,nz+1,10):
                ax.quiver(i,j,k,fieldX[i,j,k],fieldY[i,j,k],fieldZ[i,j,k])

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
X = np.arange(nx+2) 
Y = np.arange(nx+2) 
Z = np.arange(nx+2) 

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
    Ez[half_nx,half_ny,half_nz-5:half_nx+5] = Ez[half_nx,half_ny,half_nz-5:half_nz+5] + 20*np.sin(0.5*t)
    Ex,Ey,Ez = calc_Efield(Ex,Ey,Ez,Bx,By,Bz)
    Bx,By,Bz = calc_Bfield(Ex,Ey,Ez,Bx,By,Bz)

    E = np.sqrt(np.square(Ex)+np.square(Ey)+np.square(Ez))
    E_2D = E[half_nx,:,:]

    plot2Dfield(E_2D,t)
    if t ==1:
        #plot settings
        plt.colorbar () # カラーバーの表示
        plt.xlabel('X')
        plt.ylabel('Y')
    #plot3Dvector(Ex,Ey,Ez,t)
    plt.pause(.01)

