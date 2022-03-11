#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:09:45 2021

@author: joao-valeriano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as rbs
# from fast_splines import interp2d
# from numba_interp import interp2d as nb_interp2d
from numba_interp_parallel import interp2d as nb_interp2d
from numba_interp import interp2d_periodic as nb_interp2d_periodic
from numba import njit, jit
from tqdm import tqdm
from mpi4py import MPI
import os

Psi0, d, L, mu, sigma, k, v = 2, 1, 3*np.pi, 3, 2, 1, 1

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
root = 0

nx = 500
x = np.linspace(0, 2*L, nx)[:-1]
dx = np.diff(x)[0]

ny = 500
y = np.linspace(-L, L, ny)[:-1]
dy = np.diff(y)[0]

dt = 0.01
t = np.arange(0, 2*np.pi*100+dt, dt)
nt = len(t)

dtD_ratio = 1
dtD = dt/dtD_ratio

D = 1e-5 # Diffusion coefficient
r = 10. # ratio between transport and bioactivity time-scale
eps = 1e-2 # ratio between phyto and zooplankton growth rates
K = 1. # phytoplankton carrying capacity
beta = 0.43 # phytoplankton growth rate
omega = 0.34 # zooplankton mortality
P0 = 0.053
# Pe = 0.03827
# Ze = 0.04603

Pe = P0*np.sqrt(omega/(1-omega))
Ze = beta/Pe*(1-Pe/K)*(P0**2+Pe**2)

@njit
def v_x(t, x, y, Psi0, d, L, mu, sigma, k, v):
    
    return Psi0*(1 - np.tanh(y/d)**2)/d - mu*np.exp((-y**2 - (-L + x)**2)/(2*sigma**2))/sigma**2*(k*sigma**2*np.sin(k*(-t*v + y)) + y*np.cos(k*(-t*v + y)))

@njit
def v_y(t, x, y, Psi0, d, L, mu, sigma, k, v):
    
    return mu*(x-L)*np.exp((-y**2 - (-L + x)**2)/(2*sigma**2))*np.cos(k*(-t*v + y))/sigma**2

Q = 0.5
x0 = 0.3*L
l = 0.11*L

x_, y_ = np.meshgrid(x, y, indexing="ij")
P = Pe + Q*np.exp(-((x_-x0)**2+y_**2)/l**2)
# P = Pe*np.random.normal(1, 0.1, x_.shape)
Z = np.ones(P.shape)*Ze
# Z = Ze*np.random.normal(1, 0.1, x_.shape)
# P[0,:] = P[-1,:] = P[:,0] = P[:,-1] = Pe
# Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = Ze

# plt.contourf(x, y, P.T)
# plt.show()
# plt.contourf(x, y, Z.T)
# plt.show()

def mpi_x_grid_split(x, size, rank):
    if rank != size-1:
        x_idx = x[len(x)//size*rank:len(x)//size*(rank+1)]
    
    else:
        x_idx = x[len(x)//size*rank:]
        
    return x_idx

x_idx = [i for i in range(len(x))]
x_idx_diff = x_idx[1:-1]

rank_x_idx = mpi_x_grid_split(x_idx, size, rank)
rank_x_idx_diff = mpi_x_grid_split(x_idx_diff, size, rank)

@njit
def diffusion(P, Z, P_n, Z_n):
    nx = P.shape[0]
    ny = P.shape[1]
    
    for n in range(dtD_ratio):
        for i in range(nx):
            for j in range(ny):
                P_n[i,j] += (D*dtD/dx**2*(P[(i+1)%nx,j]+P[(i-1)%nx,j]-2*P[i,j])+
                            D*dtD/dy**2*(P[i,(j+1)%ny]+P[i,(j-1)%ny]-2*P[i,j]))
                Z_n[i,j] += (D*dtD/dx**2*(Z[(i+1)%nx,j]+Z[(i-1)%nx,j]-2*Z[i,j])+
                            D*dtD/dy**2*(Z[i,(j+1)%ny]+Z[i,(j-1)%ny]-2*Z[i,j]))
                
    return P_n, Z_n
    

# @jit
def rk4_adv_reac_diff_step(x, y, x_, y_, vx, vy, P, Z, dt):
    
    # P_n = P.copy()
    # Z_n = Z.copy()

    x_dt = x_ - vx*dt
    x_dt2 = x_ - vx*dt/2
    y_dt = y_ - vy*dt
    y_dt2 = y_ - vy*dt/2
    
    # vx_dt = nb_interp2d(x_dt, y_dt, x, y, vx)
    # vx_dt2 = nb_interp2d(x_dt2, y_dt2, x, y, vx)
    # vy_dt = nb_interp2d(x_dt, y_dt, x, y, vy)
    # vy_dt2 = nb_interp2d(x_dt2, y_dt2, x, y, vy)
    
    # vx_dt = nb_interp2d_periodic(x_dt, y_dt, x, y, vx)
    # vx_dt2 = nb_interp2d_periodic(x_dt2, y_dt2, x, y, vx)
    # vy_dt = nb_interp2d_periodic(x_dt, y_dt, x, y, vy)
    # vy_dt2 = nb_interp2d_periodic(x_dt2, y_dt2, x, y, vy)
    
    # k1x = vx
    # k2x = vx_dt2 + dt*k1x/2
    # k3x = vx_dt2 + dt*k2x/2
    # k4x = vx_dt + dt*k3x
    
    # xd_ = x_ - dt*(k1x + 2*k2x + 2*k3x + k4x)/6
    
    # k1y = vy
    # k2y = vy_dt2 + dt*k1y/2
    # k3y = vy_dt2 + dt*k2y/2
    # k4y = vy_dt + dt*k3y
    
    # yd_ = y_ - dt*(k1y + 2*k2y + 2*k3y + k4y)/6

    # P_n = nb_interp2d_periodic(xd_, yd_, x, y, P)
    # Z_n = nb_interp2d_periodic(xd_, yd_, x, y, Z)

    P_n = nb_interp2d_periodic(x_dt, y_dt, x, y, P)
    Z_n = nb_interp2d_periodic(x_dt, y_dt, x, y, Z)

    # P_n = nb_interp2d(xd_, yd_, x, y, P)
    # Z_n = nb_interp2d(xd_, yd_, x, y, Z)

    # P_n = nb_interp2d(x_dt, y_dt, x, y, P)
    # Z_n = nb_interp2d(x_dt, y_dt, x, y, Z)

    P = np.copy(P_n)
    Z = np.copy(Z_n)

    k1P = r*(beta*P*(1-P/K)-P**2/(P0**2+P**2)*Z)
    k1Z = r*eps*(P**2/(P0**2+P**2)-omega)*Z
    
    k2P = r*(beta*(P+dt*k1P/2)*(1-(P+dt*k1P/2)/K)-(P+dt*k1P/2)**2/(P0**2+(P+dt*k1P/2)**2)*(Z+dt*k1Z/2))
    k2Z = r*eps*((P+dt*k1P/2)**2/(P0**2+(P+dt*k1P/2)**2)-omega)*(Z+dt*k1Z/2)
    
    k3P = r*(beta*(P+dt*k2P/2)*(1-(P+dt*k2P/2)/K)-(P+dt*k2P/2)**2/(P0**2+(P+dt*k2P/2)**2)*(Z+dt*k2Z/2))
    k3Z = r*eps*((P+dt*k2P/2)**2/(P0**2+(P+dt*k2P/2)**2)-omega)*(Z+dt*k2Z/2)
    
    k4P = r*(beta*(P+dt*k3P)*(1-(P+dt*k3P)/K)-(P+dt*k3P)**2/(P0**2+(P+dt*k3P)**2)*(Z+dt*k3Z))
    k4Z = r*eps*((P+dt*k3P)**2/(P0**2+(P+dt*k3P)**2)-omega)*(Z+dt*k3Z)
    
    P_reac = dt/6 * (k1P + 2*k2P + 2*k3P + k4P)
    Z_reac = dt/6 * (k1Z + 2*k2Z + 2*k3Z + k4Z)
    
    # for i in range(dtD_ratio):
 	  #   P_n[1:-1,1:-1] += (D*dtD/dx**2*(P[2:,1:-1]+P[:-2,1:-1]-2*P[1:-1,1:-1])
 	  #                     + D*dtD/dy**2*(P[1:-1,2:]+P[1:-1,:-2]-2*P[1:-1,1:-1]))
 	    
 	  #   Z_n[1:-1,1:-1] += (D*dtD/dx**2*(Z[2:,1:-1]+Z[:-2,1:-1]-2*Z[1:-1,1:-1])
 	  #                     + D*dtD/dy**2*(Z[1:-1,2:]+Z[1:-1,:-2]-2*Z[1:-1,1:-1]))
    
    # nx = P.shape[0]
    # ny = P.shape[1]
    
    # for n in range(dtD_ratio):
    #     for i in range(nx):
    #         for j in range(ny):
    #             P_n[i,j] += (D*dtD/dx**2*(P[(i+1)%nx,j]+P[(i-1)%nx,j]-2*P[i,j])+
    #                         D*dtD/dy**2*(P[i,(j+1)%ny]+P[i,(j-1)%ny]-2*P[i,j]))
    #             Z_n[i,j] += (D*dtD/dx**2*(Z[(i+1)%nx,j]+Z[(i-1)%nx,j]-2*Z[i,j])+
    #                         D*dtD/dy**2*(Z[i,(j+1)%ny]+Z[i,(j-1)%ny]-2*Z[i,j]))
    
    P_n, Z_n = diffusion(P, Z, P_n, Z_n)
    
    P = P_n + P_reac
    Z = Z_n + Z_reac

    return P, Z

P_conc = []
Z_conc = []
t_list = []

Pmax = 0
Zmax = 0

# P_data = open("adv_reac_diff_HG_gif_nx_500_dt_0.1/P_data.txt", "w")
# Z_data = open("adv_reac_diff_HG_gif_nx_500_dt_0.1/Z_data.txt", "w")

# vx = np.zeros_like(x_)
# vy = np.copy(vx)

for n in tqdm(range(nt)):
    
    vx = v_x(t[n-1], x_, y_, Psi0, d, L, mu, sigma, k, v)
    vy = v_y(t[n-1], x_, y_, Psi0, d, L, mu, sigma, k, v)
    
    # P_rank, Z_rank = rk4_adv_reac_diff_step(x, y, x_, y_, vx, vy, P, Z, dt, rank_x_idx)
    P, Z = rk4_adv_reac_diff_step(x, y, x_, y_, vx, vy, P, Z, dt)
    
    P[P<0] = 0.
    Z[Z<0] = 0.
    
    # P[0,:] = P[-1,:] = P[:,0] = P[:,-1] = Pe
    # Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = Ze
    
    # P[:,0] = P[:,-1] = Pe
    # Z[:,0] = Z[:,-1] = Ze

    # P_root = comm.gather(P_rank, root)
    # Z_root = comm.gather(Z_rank, root)
    
    # if rank == root:
    #     P = np.concatenate(P_root)
    #     Z = np.concatenate(Z_root)
        
    #     P[P<0] = 0.
    #     Z[Z<0] = 0.
        
    #     P[0,:] = P[-1,:] = P[:,0] = P[:,-1] = Pe
    #     Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = Ze
        
    #     wait_var = 1
    # else:
    #     wait_var = 0
        
    # wait_var = comm.bcast(wait_var, root)
    # P = comm.bcast(P, root)
    # Z = comm.bcast(Z, root)
    
    if n % 100 == 0 and rank == root:

        # np.savetxt(P_data, P)
        # np.savetxt(Z_data, Z)

        # if np.max(P) > Pmax:
        #     Pmax = np.max(P)
        # if np.max(Z) > Zmax:
        #     Zmax = np.max(Z)
        # print(f"P_max={np.max(P)}; Z_max={np.max(Z)}")
        P_conc.append(np.mean(P)*dx*dy)
        Z_conc.append(np.mean(Z)*dx*dy)
        t_list.append(t[n]/(2*np.pi))
        
        # plt.figure(figsize=(5,5))
        # # plt.subplot(1,2,1)
        # plt.contourf(x, y, P.T, alpha=0.5, levels=np.linspace(0, 1, 50), cmap="Reds", antialiased=True)
        # # plt.colorbar(pad=0.01)
        # plt.contourf(x, y, Z.T, alpha=0.5, levels=np.linspace(Ze, 0.2, 50), cmap="Greens", antialiased=True)
        # plt.axis("off")
        # # plt.colorbar(pad=0.01)
        # # plt.subplot(1,2,2)
        # plt.plot(t_list, P_conc, c="red")
        # plt.plot(t_list, Z_conc, c="green")
        # plt.show()
        # plt.savefig(f"adv_reac_diff_HG_gif_nx_500_dt_0.1/ard{n//5:04d}.png", bbox_inches="tight")
        # plt.close()
        
# print(Pmax, Zmax)
    
# concentrations = np.concatenate((t_list, P_conc, Z_conc)).reshape((3,len(t_list))).T

# np.savetxt("adv_reac_diff_HG_gif_nx_500_dt_0.001/concentrations.txt", concentrations)