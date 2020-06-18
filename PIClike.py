#coding: UTF-8
import math
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def calc_k1(p,func,h):
    return h*func(p)

def RungeKutta(p,func,h):
    k_1 = calc_k1(p,func,h)
    k_2 = calc_k1(p+0.5*h*k_1,func,h)
    k_3 = calc_k1(p+0.5*h*k_2,func,h)
    k_4 = calc_k1(p+h*k_3,func,h)
    k=(k_1+2*k_2+2*k_3+k_4)/6
    return h*k

def interpolation(x,Ex,Ey,Ez,Bx,By,Bz):
    E_ip = np.zeros(3)
    B_ip = np.zeros(3)
    #integer
    ifx = math.floor(x[0]+2)-1
    ify = math.floor(x[1]+2)-1
    ifz = math.floor(x[2]+2)-1
    ihx = math.floor(x[0]+1.5)-1
    ihy = math.floor(x[1]+1.5)-1
    ihz = math.floor(x[2]+1.5)-1
    #
    sfx2 = x[0]-ifx
    sfx1 = 1-sfx2
    sfy2 = x[1]-ify
    sfy1 = 1-sfy2
    sfz2 = x[2]-ifz
    sfz1 = 1-sfz2
    
    shx2 = x[0]-ihx
    shx1 = 1-shx2
    shy2 = x[1]-ihy
    shy1 = 1-shy2
    shz2 = x[2]-ihz
    shz1 = 1-shz2

    E_ip[0] = shx1*sfy1*sfz1*Ex[ihx,ify,ifz]   + shx2*sfy1*sfz1*Ex[ihx+1,ify,ifz]   + shx2*sfy2*sfz1*Ex[ihx+1,ify+1,ifz]   + shx1*sfy2*sfz1*Ex[ihx,ify+1,ifz]\
             +shx1*sfy1*sfz2*Ex[ihx,ify,ifz+1] + shx2*sfy1*sfz2*Ex[ihx+1,ify,ifz+1] + shx2*sfy2*sfz2*Ex[ihx+1,ify+1,ifz+1] + shx1*sfy2*sfz2*Ex[ihx,ify+1,ifz+1]
    E_ip[1] = sfx1*shy1*sfz1*Ey[ifx,ihy,ifz]   + sfx2*shy1*sfz1*Ey[ifx+1,ihy,ifz]   + sfx2*shy2*sfz1*Ey[ifx+1,ihy+1,ifz]   + sfx1*shy2*sfz1*Ey[ifx,ihy+1,ifz]\
             +sfx1*shy1*sfz2*Ey[ifx,ihy,ifz+1] + sfx2*shy1*sfz2*Ey[ifx+1,ihy,ifz+1] + sfx2*shy2*sfz2*Ey[ifx+1,ihy+1,ifz+1] + sfx1*shy2*sfz2*Ey[ifx,ihy+1,ifz+1]
    E_ip[2] = sfx1*sfy1*shz1*Ez[ifx,ify,ihz]   + sfx2*sfy1*shz1*Ez[ifx+1,ify,ihz]   + sfx2*sfy2*shz1*Ez[ifx+1,ify+1,ihz]   + sfx1*sfy2*shz1*Ez[ifx,ify+1,ihz]\
             +sfx1*sfy1*shz2*Ez[ifx,ify,ihz+1] + sfx2*sfy1*shz2*Ez[ifx+1,ify,ihz+1] + sfx2*sfy2*shz2*Ez[ifx+1,ify+1,ihz+1] + sfx1*sfy2*shz2*Ez[ifx,ify+1,ihz+1]
    
    B_ip[0] = sfx1*shy1*shz1*Bx[ifx,ihy,ihz]   + sfx2*shy1*shz1*Bx[ifx+1,ihy,ihz]   + sfx2*shy2*shz1*Bx[ifx+1,ihy+1,ihz]   + sfx1*shy2*shz1*Bx[ifx,ihy+1,ihz]\
             +sfx1*shy1*shz2*Bx[ifx,ihy,ihz+1] + sfx2*shy1*shz2*Bx[ifx+1,ihy,ihz+1] + sfx2*shy2*shz2*Bx[ifx+1,ihy+1,ihz+1] + sfx1*shy2*shz2*Bx[ifx,ihy+1,ihz+1]
    B_ip[1] = shx1*sfy1*shz1*By[ihx,ify,ihz]   + shx2*sfy1*shz1*By[ihx+1,ify,ihz]   + shx2*sfy2*shz1*By[ihx+1,ify+1,ihz]   + shx1*sfy2*shz1*By[ihx,ify+1,ihz]\
             +shx1*sfy1*shz2*By[ihx,ify,ihz+1] + shx2*sfy1*shz2*By[ihx+1,ify,ihz+1] + shx2*sfy2*shz2*By[ihx+1,ify+1,ihz+1] + shx1*sfy2*shz2*By[ihx,ify+1,ihz+1]
    B_ip[2] = shx1*shy1*sfz1*Bz[ihx,ihy,ifz]   + shx2*shy1*sfz1*Bz[ihx+1,ihy,ifz]   + shx2*shy2*sfz1*Bz[ihx+1,ihy+1,ifz]   + shx1*shy2*sfz1*Bz[ihx,ihy+1,ifz]\
             +shx1*shy1*sfz2*Bz[ihx,ihy,ifz+1] + shx2*shy1*sfz2*Bz[ihx+1,ihy,ifz+1] + shx2*shy2*sfz2*Bz[ihx+1,ihy+1,ifz+1] + shx1*shy2*sfz2*Bz[ihx,ihy+1,ifz+1]
    return E_ip,B_ip
   
def Buneman_Boris(x,v,Ex,Ey,Ez,Bx,By,Bz):
    #initialization
    v_plus = np.zeros(3)
    v_0    = np.zeros(3)
    E_ip = np.zeros(3)
    B_ip = np.zeros(3)
    E_ip,B_ip = interpolation(x,Ex,Ey,Ez,Bx,By,Bz)
    alpha = qm*0.5*dt
    beta  = 2/(1+alpha*alpha*(B_ip[0]**2+B_ip[1]**2+B_ip[2]**2))
    #step1
    v_minus = v + alpha*E_ip
    #step2
    v_0[0] = v_minus[0] + alpha*(v_minus[1]*B_ip[2] - v_minus[2]*B_ip[1])
    v_0[1] = v_minus[1] + alpha*(v_minus[2]*B_ip[0] - v_minus[0]*B_ip[2])
    v_0[2] = v_minus[2] + alpha*(v_minus[0]*B_ip[1] - v_minus[1]*B_ip[0])
    #step3
    v_plus[0] = v_minus[0] + alpha*beta*(v_0[1]*B_ip[2] - v_0[2]*B_ip[1])
    v_plus[1] = v_minus[1] + alpha*beta*(v_0[2]*B_ip[0] - v_0[0]*B_ip[2])
    v_plus[2] = v_minus[2] + alpha*beta*(v_0[0]*B_ip[1] - v_0[1]*B_ip[0])
    #step4
    v_new = v_plus + alpha*E_ip
    return v_new

def calc_Position(x,v):
    x_new = x+v*dt
    if x_new[0] >= nx:
        x_new[0] = x_new[0]%nx
    if x_new[0] < 0:
        x_new[0] = -1*(abs(x_new[0])%nx)+nx
    if x_new[1] >= ny:
        x_new[1] = x_new[1]%ny
    if x_new[1] < 0:
        x_new[1] = -1*(abs(x_new[1])%ny)+ny
    if x_new[2] >= nz:
        x_new[2] = x_new[2]%nz
    if x_new[2] < 0:
        x_new[2] = -1*(abs(x_new[2])%nz)+nz
    return x_new

def dipole_antenna(Ez,t,amp,freq):
    #Ez[half_nx,half_ny,half_nz] = Ez[half_nx,half_ny,half_nz] + amp*np.sin(freq*t)
    Jz[half_nx,half_ny+5,half_nz] =  amp*np.sin(freq*t)
    Jz[half_nx,half_ny-5,half_nz] =  amp*np.sin(freq*t)

def plot2Dfield(field2D,t):
    plt.imshow(field2D,cmap='plasma')
    plt.gca().invert_yaxis()
    plt.colorbar() # カラーバーの表示
    plt.xlabel('X [dx]')
    plt.ylabel('Y [dy]')
    #time_now = "Time: " + str(round(t*dt,2)) + "[dt]"
    time_now = "Time: " + str(t) + " [dt]"
    plt.title(time_now)
    plt.pause(.01)
    plt.clf()
    
def plot3Dmotion(x_save):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x_save[0,:],x_save[1,:],x_save[2,:])
    #ax.set_xlim(0, nx)
    #ax.set_ylim(0, ny)
    #ax.set_zlim(0, nz)
    plt.show()    

#initial settings
nx = 200
ny = 200
nz = 200
half_nx = int(nx/2)
half_ny = int(ny/2)
half_nz = int(nz/2)
c = 20
dt = 0.02
dx = 1
dy = 1
dz = 1
qm = -1
inv_dt = 1/dt
inv_dx = 1/dx 
inv_dy = 1/dy
inv_dz = 1/dz
simulation_time = 1000
X = np.arange(nx+2) 
Y = np.arange(nx+2) 
Z = np.arange(nx+2) 

amp = 1
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
x = np.zeros(3)
v = np.zeros(3)

x_save = np.zeros((3,simulation_time))
v_save = np.zeros((3,simulation_time))

#Bz = np.ones((nx+2,ny+2,nz+2))
#Bz = Bz*10
#Ez = np.ones((nx+2,ny+2,nz+2))
#Ez = Ez*10
x = [half_nx,half_ny,half_nz]
#v = [0, 1, 0]
for t in range(simulation_time):
    print(t)
    dipole_antenna(Ez,t,amp,0.5)
    Ex,Ey,Ez = calc_Efield(Ex,Ey,Ez,Bx,By,Bz)
    Bx,By,Bz = calc_Bfield(Ex,Ey,Ez,Bx,By,Bz)
    v        = Buneman_Boris(x,v,Ex,Ey,Ez,Bx,By,Bz)
    x        = calc_Position(x,v) 
    E = np.sqrt(np.square(Ex)+np.square(Ey)+np.square(Ez))
    E_2D = E[:,half_ny,:]

    #plot2Dfield(E_2D,t)
    x_save[:,t] = x
    v_save[:,t] = v

plot3Dmotion(x_save)

