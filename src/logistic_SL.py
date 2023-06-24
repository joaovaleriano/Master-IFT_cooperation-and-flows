#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:22:49 2021

@author: joao-valeriano
"""

#%% Packages

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import numba_interp_parallel as nb_i_p
from tqdm import tqdm
import os
import h5py


#%%

@njit(parallel=True)
def pv_field_domain(x_, y_, pvs, strengths, bounds, periodic_repeats=2):
    # (x_, y_): coordinates of meshgrid where to calculate the velocity field
    # pvs: coordinates of point-vortices
    # strengths: vector with rotation strength \Gamma of each point-vortex
    # bounds: bounds of a side of the domain
    # periodic_repeats: number of periodic repetitions of the domain to consider for
    #                   implementing periodic boundary conditions on the flow
        
    L = bounds[1]-bounds[0]
    
    vx = np.zeros_like(x_)
    vy = np.zeros_like(x_)
    
    for n in range(len(pvs)):
        for j in prange(x_.shape[0]):
            for k in prange(y_.shape[1]):
                vx_temp = 0
                vy_temp = 0
                for i in range(-periodic_repeats, periodic_repeats+1):
                    x = x_[j,k]
                    y = y_[j,k]
                    
                    dx = x-pvs[n,0]
                    dy = y-pvs[n,1]
                    
                    if dx-i*L != 0 or dy != 0:
                        vx_temp -= np.sin(2*np.pi*dy/L)/(np.cosh(2*np.pi*dx/L-2*np.pi*i)-np.cos(2*np.pi*dy/L))
                    if dx != 0 or dy-i*L != 0:
                        vy_temp += np.sin(2*np.pi*dx/L)/(np.cosh(2*np.pi*dy/L-2*np.pi*i)-np.cos(2*np.pi*dx/L))
                        
                vx[j,k] += strengths[n]/(2*L)*vx_temp
                vy[j,k] += strengths[n]/(2*L)*vy_temp
        
    return vx, vy


@njit
def pv_field_vort(pvs, strengths, bounds, periodic_repeats=2):
    # pvs: coordinates of point-vortices
    # strengths: vector with rotation strength \Gamma of each point-vortex
    # bounds: bounds of a side of the domain
    # periodic_repeats: number of periodic repetitions of the domain to consider for
    #                   implementing periodic boundary conditions on the flow
        
        
    L = bounds[1]-bounds[0]
    
    v = np.zeros_like(pvs)
    
    for n in range(len(pvs)):
        for m in range(len(pvs)):
            x, y = pvs[n]
            for i in range(-periodic_repeats, periodic_repeats+1):
                dx = x-pvs[m,0]
                dy = y-pvs[m,1]
                
                if dx-i*L != 0 or dy != 0:
                    v[n,0] -= np.sin(2*np.pi*dy/L)/(np.cosh(2*np.pi*dx/L-2*np.pi*i)-np.cos(2*np.pi*dy/L))
                if dx != 0 or dy-i*L != 0:
                    v[n,1] += np.sin(2*np.pi*dx/L)/(np.cosh(2*np.pi*dy/L-2*np.pi*i)-np.cos(2*np.pi*dx/L))
            
        v[n] *= strengths[n]/(2*L)
    
    return v


# Ranking vortex velocity field
@njit
def rankine_field(xc, yc, x_, y_, gamma, a):
    # (xc, yc): coordinates of the vortex center
    # (x_, y_): coordinates of the meshgrid where to calculate the field
    # gamma: vortex rotation speed multiplying factor
    # a: vortex radius
    
    vx = np.ones_like(x_)*gamma/(2*np.pi)
    vy = np.ones_like(y_)*gamma/(2*np.pi)
    
    r = np.sqrt((x_-xc)**2+(y_-yc)**2)
    
    for i in range(x_.shape[0]):
        for j in range(x_.shape[1]):
            if r[i,j] <= a:
                v = r[i,j]/a**2
            else:
                v = 1/r[i,j]
                
            vx[i,j] *= -v*(y_[i,j]-yc)/r[i,j]
            vy[i,j] *= v*(x_[i,j]-xc)/r[i,j]
            
    return vx, vy


# Euler integration of the advection-reaction-diffusion system
@jit(parallel=True)
def euler_adv_reac_step_single_species(x, y, x_, y_, vx, vy, gamma, D, u, dt, dtD_ratio, dtD, m2):
    
    # Reproduction and non-local competition (convolution via FFT)
    u += dt * r*u*(1-1/K*dx*dy*fftpack.ifftshift(fftpack.ifft2(fftpack.fft2(u)*fftpack.fft2(m2)).astype(np.float64)))
    
    # Finite difference diffusion
    for i in range(dtD_ratio):
        u2 = np.copy(u)
        for j in prange(nx):
            for k in prange(ny):                
                u[j,k] += (D*dtD/dx**2*(u2[(j+1)%nx,k]+u2[(j-1)%nx,k]-2*u2[j,k]) + 
                                D*dtD/dy**2*(u2[j,(k+1)%ny]+u2[j,(k-1)%ny]-2*u2[j,k]))
    
    # Advection - Semi-Lagrangian integration
    x_dt = x_ - gamma*vx*dt
    y_dt = y_ - gamma*vy*dt
    u_n = nb_i_p.interp2d_periodic(x_dt, y_dt, x, y, u, rescale=False)

    return u_n

#%%

# Seed for reproducibility of initial condition
seed = 3
np.random.seed(seed)

# Domain definition
bounds = np.array([-np.pi, np.pi])
L = bounds[1]-bounds[0]

nx = 255
dx = L/nx
x = np.linspace(*bounds, nx+1)[1:]

ny = 255
dy = L/ny
y = np.linspace(*bounds, ny+1)[1:]

x_, y_ = np.meshgrid(x, y, indexing="ij")

# Biological parameters
D = 1e-3 # Diffusion coeff.
r = 2 # Birth rate
K = 1 # Carrying capacity
comp_rad = 1. # Competition radius

gamma = [0.1] # List of velocity multipliers to consider


#%% Non-local competition kernel

r_int = int(comp_rad/dx)

m = np.zeros((1+2*r_int,1+2*r_int))
m_norm = 0
for i in range(-r_int,r_int+1):
    for j in range(-r_int,r_int+1):
        if i**2+j**2<=r_int**2:
            m[i+r_int,j+r_int] = 1.
            m_norm += 1

m2 = np.zeros((nx,ny))
m2[nx//2-r_int:nx//2+r_int+1,ny//2-r_int:ny//2+r_int+1] = m


#%% Simulation Parameters

u0 = np.random.normal(1, 0.5, size=x_.shape)
u = np.copy(u0)

# Point-vortex field
# n_vortex = 5
# space_repetitions = 2
# strengths = np.random.choice([-1, 1], size=n_vortex)
# pvs = np.random.uniform(-np.pi, np.pi, (n_vortex, 2))
# vx, vy = pv_field_domain(x_, y_, pvs, strengths, bounds, space_repetitions)

# Rankine vortex field
# vx, vy = rankine_field(0, 0, x_, y_, 1, 0.5)

# Sinusoidal flow
vx = np.sin(y_)
vy = np.zeros_like(vx)


#%% Simulation

n_plots = 1000 # number of plots to save through simulation

# Create folder to save results
if "logistic_SL_results" not in os.listdir():
    os.mkdir("logistic_SL_results")
    
# Loop over different values for the velocity multiplying factor gamma
for g in gamma:

    # Create folder for results
    # path = f"logistic_SL_results/gamma{g:.6f}_D{D:.5f}_R{comp_rad:.6f}_seed{seed}"
    # if path.split("/")[1] not in os.listdir("logistic_SL_results"):
    #     os.mkdir(path)
        
    # Create file for results
    # h5file = h5py.File(f"{path}/dat.h5", "w")
    
    # Initializations
    np.random.seed(seed)
    u = np.random.normal(1, 0.1, size=x_.shape)
    
    T = 1500 # simulation duration
    # dt = 0.01
    dt = min(0.01, (dx*dx+dy*dy)/D/8)
    if g != 0:
        dt = min(dt, 1/(np.max(np.abs(g*vx))/dx+np.max(np.abs(g*vy))/dy)) # step based on flow velocity
    dtD_ratio = max(1, round(dt/((dy**2+dx**2)/D/8)))
    dtD = dt/dtD_ratio
    t = np.arange(0, T+dt, dt)
    nt = len(t)
    
    conc_time = [t[0]] # Time vector
    conc = [np.mean(u)] # Avg concentration over time

    # Plot - Initial Condition
    plt.subplots(1, 2, figsize=(25,10))
    plt.subplots_adjust(wspace=0.05)
    plt.subplot(1,2,1)
    plt.imshow(u.T, cmap="Greens", origin="lower", extent=np.concatenate((bounds, bounds)))
    plt.colorbar(ticks=np.linspace(np.min(u), np.min(u)+0.9*(np.max(u)-np.min(u)), 7))
    # plt.scatter(*pvs[strengths>0].T, c="b", s=20) # Plot vortices' positions
    # plt.scatter(*pvs[strengths<0].T, c="r", s=20) # Plot vortices' positions
    plt.title(f"t = {t[0]:0.1f}")
    plt.subplot(1,2,2)
    plt.plot(conc_time, conc, c="g")

    # Choose to show plot live or save
    plt.show()
    # plt.savefig(f"point_vortex_movie/single_vortex{0:05d}")
    plt.close()
    
    for n in tqdm(range(1, nt)):
        u = euler_adv_reac_step_single_species(x, y, x_, y_, vx, vy, g, D, u, dt, dtD_ratio, dtD, m2)
        u[u < 0.] = 0. # Avoid negative abundances associated with numerical error
        
        # Update point-vortices' positions -- comment if using other flow
        # pvs += dt*g*pv_field_vort(pvs, strengths, bounds, space_repetitions)
        # pvs = (pvs-bounds[0])%L + bounds[0]
        # vx, vy = pv_field_domain(x_, y_, pvs, strengths, bounds, space_repetitions)
    
        if n % (nt//n_plots) == 0:
            
            # Save total concentration
            conc_time.append(t[n])
            conc.append(np.mean(u))
    
            # Plot
            plt.subplots(1, 2, figsize=(25,10))
            plt.subplots_adjust(wspace=0.05)
            plt.subplot(1,2,1)
            plt.imshow(u.T, cmap="Greens", origin="lower", extent=np.concatenate((bounds, bounds)))
            plt.colorbar(ticks=np.linspace(np.min(u), np.min(u)+0.9*(np.max(u)-np.min(u)), 7))
            # plt.scatter(*pvs[strengths>0].T, c="b", s=20) # Plot vortices' positions
            # plt.scatter(*pvs[strengths<0].T, c="r", s=20) # Plot vortices' positions
            plt.title(f"t = {t[n]:0.1f}")
            plt.subplot(1,2,2)
            plt.plot(conc_time, conc, c="g")
            
            # Choose to show plot live or save
            plt.show()
            # plt.savefig(f"{path}/fig{n // (nt//n_plots):03d}")
            plt.close()
            
    #         h5file.create_dataset(f"t{t[n]}", data=u) # save concentration
    
    # save results
    # h5file.create_dataset("time", data=conc_time)
    # h5file.create_dataset("conc", data=conc)
    # h5file.close()