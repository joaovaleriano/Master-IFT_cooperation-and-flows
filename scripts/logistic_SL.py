#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:22:49 2021

@author: joao-valeriano
"""

import numpy as np
from scipy import fft, fftpack
from scipy.integrate import simps
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads
from numba_interp import interp2d as nb_interp2d
from numba_interp import interp2d_periodic as nb_interp2d_periodic
import numba_interp_parallel as nb_i_p
from smooth_gaussian import smooth_gaussian
from tqdm import tqdm
import time
import os
import h5py

set_num_threads(6)

@njit(fastmath=True)
def rk4_step(f, y, t, args, h=0.01):
    # f: function to be integrated; y0: initial conditions;
    # t: time points for the function to be evaluated;
    # args: extra function parameters;
    # h: time step

    k1 = f(t, y, args)
    k2 = f(t + h / 2., y + k1 * h / 2., args)
    k3 = f(t + h / 2., y + k2 * h / 2., args)
    k4 = f(t + h, y + k3 * h, args)
    y_ = y + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    
    return y_

@njit(fastmath=True, parallel=True)
def pv_field(t, vortices, strengths, x_, y_):
    
    vx = np.zeros_like(x_)
    vy = np.zeros_like(y_)
    
    for i in prange(len(vortices)):
        # vx -= strengths[i] * (y_-vortices[i,1]) / ((x_-vortices[i,0])**2+(y_-vortices[i,1])**2)
        # vy += strengths[i] * (x_-vortices[i,0]) / ((x_-vortices[i,0])**2+(y_-vortices[i,1])**2)
        for j in prange(x_.shape[0]):
            for k in prange(x_.shape[1]):
                vx[j,k] -= strengths[i] * (y_[j,k]-vortices[i,1]) / ((x_[j,k]-vortices[i,0])**2+(y_[j,k]-vortices[i,1])**2)
                vy[j,k] += strengths[i] * (x_[j,k]-vortices[i,0]) / ((x_[j,k]-vortices[i,0])**2+(y_[j,k]-vortices[i,1])**2)
        
    vx /= 2*np.pi
    vy /= 2*np.pi
    
    return vx, vy


@njit(fastmath=True, parallel=True)
def pv_model(t, vortex_particle_pos, strengths):

    v = np.zeros_like(vortex_particle_pos)

    vortex_pos = vortex_particle_pos[:len(strengths)]

    for i in prange(len(vortex_particle_pos)):
        for j in prange(len(strengths)):
            if (vortex_particle_pos[i,0]-vortex_pos[j,0])**2+(vortex_particle_pos[i,1]-vortex_pos[j,1])**2 != 0:
                v[i,0] -= strengths[j]*(vortex_particle_pos[i,1]-vortex_pos[j,1])/((vortex_particle_pos[i,0]-vortex_pos[j,0])**2+(vortex_particle_pos[i,1]-vortex_pos[j,1])**2)
                v[i,1] += strengths[j]*(vortex_particle_pos[i,0]-vortex_pos[j,0])/((vortex_particle_pos[i,0]-vortex_pos[j,0])**2+(vortex_particle_pos[i,1]-vortex_pos[j,1])**2)
    
    v /= 2*np.pi
    
    return v


@njit
def pv_vorticity_field(vortices, strengths, x_, y_):
    
    vorticity = np.zeros_like(x_)
    
    for i in range(len(vortices)):
        vorticity -= strengths[i] / ((x_-vortices[i,0])**2+(y_-vortices[i,1])**2)
        
    vorticity /= 2*np.pi
    
    return vorticity

@njit
def fix_periodic_bounds(pos, bounds):
    pos[:,0] -= (bounds[0,1]-bounds[0,0])*(np.sign(pos[:,0]-bounds[0,1])+1)/2
    pos[:,0] -= (bounds[0,1]-bounds[0,0])*(np.sign(pos[:,0]-bounds[0,0])-1)/2
    pos[:,1] -= (bounds[1,1]-bounds[1,0])*(np.sign(pos[:,1]-bounds[1,1])+1)/2
    pos[:,1] -= (bounds[1,1]-bounds[1,0])*(np.sign(pos[:,1]-bounds[1,0])-1)/2
    
    return pos

@njit(fastmath=True)#, parallel=True)
def periodic_repetitions(pos, bounds, repetitions):
    reps = 1+2*repetitions
    new_pos = np.zeros((len(pos)*reps**2, 2))
    new_pos[:len(pos)] = pos
    
    idx = len(pos)
    
    for n in prange(reps):
        for m in prange(reps):
            if n != repetitions or m != repetitions:
                for i in prange(len(pos)):
                    new_pos[idx,0] = pos[i,0]+(bounds[0,1]-bounds[0,0])*(n-repetitions)
                    new_pos[idx,1] = pos[i,1]+(bounds[1,1]-bounds[1,0])*(m-repetitions)
                    
                    idx += 1

    return new_pos

@njit
def diff2(u, D, dt, dx):
    
    n = len(u)
    f = np.zeros(n)
    
    for i in range(n):
        f[i] = u[i] + D*dt/dx**2*(u[(i+1)%n]+u[(i-1)%n]-2*u[i])

    return f

@njit
def rankine_field(xc, yc, x_, y_, gamma, a):
    
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


np.random.seed(21314123)

bounds_x = np.array([-np.pi, np.pi])
length_x = bounds_x[1]-bounds_x[0]
nx = 255
dx = length_x/nx
x = np.linspace(*bounds_x, nx+1)[1:]
# x = np.arange(*bounds_x, dx)

bounds_y = np.array([-np.pi, np.pi])
length_y = bounds_y[1]-bounds_y[0]
ny = 255
dy = length_y/ny
y = np.linspace(*bounds_y, ny+1)[1:]
# y = np.arange(*bounds_y, dy)pl

x_, y_ = np.meshgrid(x, y, indexing="ij")

bounds = np.array([bounds_x, bounds_y])

comp_rad = 1.
D = 1e-3
r = 2
K = 1
gamma = 0

r_int_r = 0
# comp_rad = 0.1
r_int = int(comp_rad/dx)

int_idx_r = []
for i in range(-r_int_r, r_int_r+1):
    for j in range(-r_int_r, r_int_r+1):
        if i**2+j**2 <= r_int_r**2:
            int_idx_r.append([i,j])
            # plt.scatter(x[i+r], y[j+r])
int_idx_r = np.array(int_idx_r)
int_area_r = len(int_idx_r)#*dx*dy

int_idx = []
for i in range(-r_int, r_int+1):
    for j in range(-r_int, r_int+1):
        if i**2+j**2 <= r_int**2:
            int_idx.append([i,j])
            # plt.scatter(x[i+r], y[j+r])
int_idx = np.array(int_idx)
int_area = len(int_idx)#*dx*dy

m = np.zeros((1+2*r_int,1+2*r_int))
m_norm = 0
for i in range(-r_int,r_int+1):
    for j in range(-r_int,r_int+1):
        if i**2+j**2<=r_int**2:
            m[i+r_int,j+r_int] = 1.
            m_norm += 1

# m *= dx*dx/(np.pi*comp_rad**2)

m2 = np.zeros((nx,ny))
m2[nx//2-r_int:nx//2+r_int+1,ny//2-r_int:ny//2+r_int+1] = m

# u0 = np.zeros_like(x_)
# u0[(x_-1)**2+y_**2 < (np.pi/4)**2] = K/10
u0 = np.random.normal(1, 0.5, size=x_.shape)
# u0 = np.random.uniform(1, 0.1, size=x_.shape)
# u0 = K * (1 + 0.1*np.sin(5*x_)*np.sin(5*y_))
# u0[x_**2+y_**2 > (0.9*np.pi)**2] = 0
# u0[0] = K
# u0 = np.ones_like(x_)*K
# u0 = np.exp(-((x_)**2+(y_)**2)/(2*0.5))
# u0 = np.ones_like(x_)*0.5 + np.random.normal(scale=0.1, size=x_.shape)
# u0[:50,:100] = u0[:50,-100:] = 1.
# u0 = np.exp(-((x_-1)**2+y_**2)/(2*0.5))
u = np.copy(u0)

n_vortex = 7
n_particle = 0
space_repetitions = 3

# strengths = np.ones(2)

# strengths = np.random.choice([-1, 1], size=n_vortex)
strengths = np.array([-1.]*(n_vortex//2)+[1.]*(n_vortex//2)+[np.random.choice([-1.,1.])]*(n_vortex%2))
periodic_strengths = np.array(list(strengths)*(1+2*space_repetitions)**2)

# vortices = np.array([[-1, 1],[1, -1]])
vortices = np.random.uniform(-np.pi, np.pi, (n_vortex,2))
periodic_vortices = periodic_repetitions(vortices, bounds, space_repetitions)
particles = np.random.normal(1, 0.2, size=(n_particle,2))

# vx, vy = pv_field(0, vortices, strengths, x_, y_)
vx = 10.*np.sin(y_)#+1e-2
# vx = np.sin(x_+np.pi/2)*5+5.1
# vx = -(y_-np.pi)*(y_+np.pi)/np.pi**2
# vx = 0.9/(2*np.pi)*x_
# vx[128:-128] = 1.
# vx = 1*np.ones_like(x_)
vy = vx*0
# vx *= 0
bounds = np.array([bounds_x, bounds_y])

# vx, vy = rankine_field(0, 0, x_, y_, 1, np.pi/4)  
# vx, vy = rankine_field(0, 0, x_, y_, 1, 1)
# vx1, vy1 = rankine_field(0, -1, x_, y_, 1, 1)
# vx += vx1; vy += vy1
# vx1, vy1 = rankine_field(1, 0, x_, y_, -1, 1)
# vx += vx1; vy += vy1
# vx1, vy1 = rankine_field(-1, 0, x_, y_, -1, 1)
# vx += vx1; vy += vy1
# vx *= 0
# vy *= 0

# vx[x_**2+y_**2 > (0.9*np.pi-np.sqrt((dx**2+dy**2)/2))**2] = 0
# vy[x_**2+y_**2 > (0.9*np.pi-np.sqrt((dx**2+dy**2)/2))**2] = 0

vx_norm = np.copy(vx)
vy_norm = np.copy(vy)

m = np.hypot(vx, vy)
v = np.sqrt(vx**2+vy**2)
vx_norm[np.where(v!=0)] = vx[np.where(v!=0)] / np.sqrt(vx**2+vy**2)[np.where(v!=0)]
vy_norm[np.where(v!=0)] = vy[np.where(v!=0)] / np.sqrt(vx**2+vy**2)[np.where(v!=0)]

vort_sign = np.sign(pv_vorticity_field(periodic_repetitions(vortices, bounds, space_repetitions), 
                                        periodic_strengths, x_, y_))

# T = 100
# # dt = 0.01
# dt = min(0.01, (dx*dx+dy*dy)/D/8)
# dt = min(dt, 0.5/(np.max(np.abs(vx))/dx+np.max(np.abs(vy))/dy))
# dtD_ratio = max(1, round(dt/((dy**2+dx**2)/D/8)))
# dtD = dt/dtD_ratio
# t = np.arange(0, T+dt, dt)
# nt = len(t)

# conc_time = [t[0]]
# conc = [np.mean(u)]
# conc_std = [np.std(u)]

# @njit
def conv_scale(a):
    b = np.copy(a)
    b[:,0] = (b[:,0]-bounds_x[0])/(bounds_x[1]-bounds_x[0])*nx
    b[:,1] = (b[:,1]-bounds_y[0])/(bounds_y[1]-bounds_y[0])*ny
    
    return b

# plt.subplots(1, 2, figsize=(25,10))
# plt.subplots_adjust(wspace=0.05)
# plt.subplot(1,2,1)
# # plt.contourf(x, y, u.T, 100, cmap="Greens")
# plt.imshow(u.T, cmap="Greens", origin="lower")
# plt.xticks(np.linspace(0, nx-1, 5), np.round(np.linspace(*bounds_x, 5), 2))
# plt.yticks(np.linspace(0, ny-1, 5), np.round(np.linspace(*bounds_y, 5), 2))
# plt.colorbar(ticks=np.linspace(np.min(u), np.min(u)+0.9*(np.max(u)-np.min(u)), 7))
# # plt.contourf(x, y, vort_sign.T, 1, cmap="coolwarm", alpha=0.2, antialiased=True)
# # plt.scatter(*conv_scale(vortices[strengths>0]).T, c="b", s=20)
# # plt.scatter(*conv_scale(vortices[strengths<0]).T, c="r", s=20)
# # plt.scatter(*particles.T, c="k", s=10, alpha=0.3)
# # plt.quiver(x[::10], y[::10], vx_norm[::10,::10].T, vy_norm[::10,::10].T, m[::10,::10].T, cmap="hot", alpha=0.4)
# plt.xlim(0, nx)
# plt.ylim(0, ny)
# # plt.xticks([])
# # plt.yticks([])
# plt.title(f"t = {t[0]:0.1f}")
# plt.subplot(1,2,2)
# plt.plot(conc_time, conc, c="g")
# plt.show()
# # plt.savefig(f"point_vortex_movie/single_vortex{0:05d}")
# # plt.close()

@njit
def rk4_adv_reac_step_single_species(x, y, x_, y_, vx, vy, r, K, D, u, dt, dtD_ratio, dtD):
    x_dt = x_ - vx*dt
    x_dt2 = x_ - vx*dt/2
    y_dt = y_ - vy*dt
    y_dt2 = y_ - vy*dt/2
    
    vx_dt = nb_interp2d(x_dt, y_dt, x, y, vx)
    vx_dt2 = nb_interp2d(x_dt2, y_dt2, x, y, vx)
    vy_dt = nb_interp2d(x_dt, y_dt, x, y, vy)
    vy_dt2 = nb_interp2d(x_dt2, y_dt2, x, y, vy)
    
    xd_ = x_ - dt*(vx+vx_dt)/2
    yd_ = y_ - dt*(vy+vy_dt)/2
    
    k1x = vx
    k2x = vx_dt2 + dt*k1x/2
    k3x = vx_dt2 + dt*k2x/2
    k4x = vx_dt + dt*k3x
    
    xd_ = x_ - dt*(k1x + 2*k2x + 2*k3x + k4x)/6
    
    k1y = vy
    k2y = vy_dt2 + dt*k1y/2
    k3y = vy_dt2 + dt*k2y/2
    k4y = vy_dt + dt*k3y
    
    yd_ = y_ - dt*(k1y + 2*k2y + 2*k3y + k4y)/6
    
    s = np.sum(u)
    
    # u_n = nb_interp2d_periodic(xd_, yd_, x, y, u, rescale=True)
    u_n = nb_interp2d(xd_, yd_, x, y, u, rescale=False)
    
    for j in range(nx):
        for k in range(ny):
            if x[j]**2+y[k]**2 > (0.9*np.pi)**2:
                u_n[j,k] = 0
    
    u_n *= s/np.sum(u_n)
    
    # k1 = r*u_n*(1-u_n/K)
    # k2 = r*(u_n+dt*k1/2)*(1-(u_n+dt*k1/2)/K)
    # k3 = r*(u_n+dt*k2/2)*(1-(u_n+dt*k2/2)/K)
    # k4 = r*(u_n+dt*k3)*(1-(u_n+dt*k3)/K)
    
    # u_n += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    for i in range(nx):
        for j in range(ny):
            if x[i]**2+y[j]**2 < (0.9*np.pi)**2:
                k1 = r*u_n[i,j]*(1-u_n[i,j]/K)
                k2 = r*(u_n[i,j]+dt*k1/2)*(1-(u_n[i,j]+dt*k1/2)/K)
                k3 = r*(u_n[i,j]+dt*k2/2)*(1-(u_n[i,j]+dt*k2/2)/K)
                k4 = r*(u_n[i,j]+dt*k3)*(1-(u_n[i,j]+dt*k3)/K)
                
                u_n[i,j] += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # u = np.copy(u_n)
    
    for i in range(dtD_ratio):
        for j in range(nx):
            for k in range(ny):
                u_n[j,k] += (D*dtD/dx**2*(u[(j+1)%nx,k]+u[(j-1)%nx,k]-2*u[j,k]) + 
                                D*dtD/dy**2*(u[j,(k+1)%ny]+u[j,(k-1)%ny]-2*u[j,k]))
                
    # for i in range(dtD_ratio):
    #     for j in range(1, nx-1):
    #         for k in range(1, ny-1):
    #             u_n[j,k] += (D*dtD/dx**2*(u[j+1,k]+u[j-1,k]-2*u[j,k]) + 
    #                             D*dtD/dy**2*(u[j,k+1]+u[j,k-1]-2*u[j,k]))
    
    # for i in range(dtD_ratio):
    #     for j in range(1, nx-1):
    #         for k in range(1, ny-1):
    #             if x[j]**2+y[k]**2 < (0.9*np.pi)**2:
    #                 u_n[j,k] += (D*dtD/dx**2*(u[j+1,k]+u[j-1,k]-2*u[j,k]) + 
    #                                 D*dtD/dy**2*(u[j,k+1]+u[j,k-1]-2*u[j,k]))
    #             else:
    #                 u_n[j,k] = 0
    
    # u_n *= s/np.sum(u_n)

    return u_n

@jit(parallel=True)
def euler_adv_reac_step_single_species(x, y, x_, y_, vx, vy, gamma, D, u, dt, dtD_ratio, dtD, int_idx_r, int_area_r, int_idx, int_area, m2):
    x_dt = x_ - gamma*vx*dt
    y_dt = y_ - gamma*vy*dt
    
    s = np.sum(u)
    
    u0 = np.copy(u)
    
    u += dt * r*u*(1-dx*dy*fftpack.ifftshift(fftpack.ifft2(fftpack.fft2(u)*fftpack.fft2(m2)).astype(np.float64)))
    
    # uf = fftpack.fft2(u0)
    # k = fftpack.fftfreq(len(x), np.diff(x)[0])
    # k2 = np.zeros((len(k), len(k)))
    
    # for i in range(len(k)):
    #     for j in range(len(k)):
    #         k2[i,j] = k[i]*k[i] + k[j]*k[j]
            
    # for i in range(dtD_ratio):
    #     uf += dt * ( -D*k2*uf )
        
    # u = fftpack.ifft2(uf).real
    
    for i in range(dtD_ratio):
        u2 = np.copy(u)
        for j in prange(nx):
            for k in prange(ny):                
                u[j,k] += (D*dtD/dx**2*(u2[(j+1)%nx,k]+u2[(j-1)%nx,k]-2*u2[j,k]) + 
                                D*dtD/dy**2*(u2[j,(k+1)%ny]+u2[j,(k-1)%ny]-2*u2[j,k]))
                
    # u_n = nb_i_p.interp2d_periodic(x_dt, y_dt, x, y, u, rescale=False)
    u_n = u.copy()

    return u_n


n_plots = 1000

# print(f"Courant number: {dt*(np.max(np.abs(vx))/dx+np.max(np.abs(vy))/dy)}")

# gamma = np.array([0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1])

gamma = [1.]

if "parabolic_videos" not in os.listdir():
    os.mkdir("parabolic_videos")
    
for g in gamma:
    
    seed = 0
    # path = f"parabolic_videos/gamma{g:.6f}_D{D:.5f}_R{comp_rad:.6f}_seed{seed}"
    # if path.split("/")[1] not in os.listdir("parabolic_videos"):
    #     os.mkdir(path)
    
    # h5file = h5py.File(f"{path}/dat.h5", "w")
    
    # np.random.seed(seed)
    # u = np.random.normal(1, 0.1, size=x_.shape)
    # u = np.random.uniform(1, 10, size=x_.shape)
    
    T = 1500
    # dt = 0.01
    dt = min(0.005, (dx*dx+dy*dy)/D/8)
    if g != 0:
        dt = min(dt, 0.5/(np.max(np.abs(g*vx))/dx+np.max(np.abs(g*vy))/dy))
    dtD_ratio = max(1, round(dt/((dy**2+dx**2)/D/8)))
    dtD = dt/dtD_ratio
    t = np.arange(0, T+dt, dt)
    nt = len(t)
    
    conc_time = [t[0]]
    conc = [np.mean(u)]
    conc_std = [np.std(u)]
    
    # plt.subplots(1, 2, figsize=(25,10))
    # plt.subplots_adjust(wspace=0.05)
    # plt.subplot(1,2,1)
    # # plt.contourf(x, y, u.T, 100, cmap="Greens")
    # plt.imshow(u.T, cmap="Greens", origin="lower")
    # plt.xticks(np.linspace(0, nx-1, 5), np.round(np.linspace(*bounds_x, 5), 2))
    # plt.yticks(np.linspace(0, ny-1, 5), np.round(np.linspace(*bounds_y, 5), 2))
    # plt.colorbar(ticks=np.linspace(np.min(u), np.min(u)+0.9*(np.max(u)-np.min(u)), 7))
    # # plt.contourf(x, y, vort_sign.T, 1, cmap="coolwarm", alpha=0.2, antialiased=True)
    # # plt.scatter(*conv_scale(vortices[strengths>0]).T, c="b", s=20)
    # # plt.scatter(*conv_scale(vortices[strengths<0]).T, c="r", s=20)
    # # plt.scatter(*particles.T, c="k", s=10, alpha=0.3)
    # # plt.quiver(x[::10], y[::10], vx_norm[::10,::10].T, vy_norm[::10,::10].T, m[::10,::10].T, cmap="hot", alpha=0.4)
    # # plt.xlim(-0.5)
    # plt.xlim(0, nx)
    # plt.ylim(0, ny)
    # # plt.xticks([])
    # # plt.yticks([])
    # plt.title(f"t = {0.0}")
    # plt.subplot(1,2,2)
    # plt.plot(conc_time, conc, c="g")
    # # plt.fill_between(conc_time, np.array(conc)-np.array(conc_std), np.array(conc)+np.array(conc_std), color="g", alpha=0.3)
    # plt.show()
    # plt.savefig(f"{path}/fig{0:03d}")
    # plt.close()
    
    for n in tqdm(range(1, nt)):
        
        # count_t = time.time()
        
        # u = rk4_adv_reac_step_single_species(x, y, x_, y_, vx, vy, r, K, D, u, dt, dtD_ratio, dtD)
        u = euler_adv_reac_step_single_species(x, y, x_, y_, vx, vy, g, D, u, dt, dtD_ratio, dtD, int_idx_r, int_area_r, int_idx, int_area, m2)
        
        u[u < 0.] = 0.
        # u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 0.
        # u[0] = u[-1] = 0.
        # u[0] = K
        
        # u[(x_**2+y_**2 > (0.9*np.pi)**2)] = 0
        
        # vortex_particle_pos = np.concatenate([periodic_vortices, particles])
        # vortex_particle_pos = rk4_step(pv_model, vortex_particle_pos, 0, periodic_strengths, dt)
        # vortices = vortex_particle_pos[:n_vortex]
        # particles = vortex_particle_pos[n_vortex*(1+2*space_repetitions)**2:]
        # vortices = fix_periodic_bounds(vortices, bounds)
        # periodic_vortices = periodic_repetitions(vortices, bounds, space_repetitions)
        # vx, vy = pv_field(0, periodic_vortices, periodic_strengths, x_, y_)
    
        if n % (nt//n_plots) == 0:
            # print(f"Courant number: {dt*(np.max(vx)/dx+np.max(vy)/dy)}")
            conc_time.append(t[n])
            conc.append(np.mean(u))
    
        #     m = np.hypot(vx, vy)
        #     # print(np.mean(m))
        #     # v = np.sqrt(vx**2+vy**2)
        #     # vx_norm[np.where(v!=0)] = vx[np.where(v!=0)] / np.sqrt(vx**2+vy**2)[np.where(v!=0)]
        #     # vy_norm[np.where(v!=0)] = vy[np.where(v!=0)] / np.sqrt(vx**2+vy**2)[np.where(v!=0)]
            plt.subplots(1, 2, figsize=(25,10))
            plt.subplots_adjust(wspace=0.05)
            plt.subplot(1,2,1)
            # plt.contourf(x, y, u.T, 100, cmap="Greens")
            plt.imshow(u.T, cmap="Greens", origin="lower")
            plt.xticks(np.linspace(0, nx-1, 5), np.round(np.linspace(*bounds_x, 5), 2))
            plt.yticks(np.linspace(0, ny-1, 5), np.round(np.linspace(*bounds_y, 5), 2))
            plt.colorbar(ticks=np.linspace(np.min(u), np.min(u)+0.9*(np.max(u)-np.min(u)), 7))
            # plt.contourf(x, y, vort_sign.T, 1, cmap="coolwarm", alpha=0.2, antialiased=True)
            # plt.scatter(*conv_scale(vortices[strengths>0]).T, c="b", s=20)
            # plt.scatter(*conv_scale(vortices[strengths<0]).T, c="r", s=20)
            # plt.scatter(*particles.T, c="k", s=10, alpha=0.3)
            # plt.quiver(x[::10], y[::10], vx_norm[::10,::10].T, vy_norm[::10,::10].T, m[::10,::10].T, cmap="hot", alpha=0.4)
            # plt.xlim(-0.5)
            plt.xlim(0, nx-1)
            plt.ylim(0, ny-1)
            # plt.xticks([])
            # plt.yticks([])
            plt.title(f"t = {t[n]:0.1f}")
            plt.subplot(1,2,2)
            plt.plot(conc_time, conc, c="g")
            # plt.fill_between(conc_time, np.array(conc)-np.array(conc_std), np.array(conc)+np.array(conc_std), color="g", alpha=0.3)
            plt.show()
            # plt.savefig(f"{path}/fig{n // (nt//n_plots):03d}")
            plt.close()
            
    #         h5file.create_dataset(f"t{t[n]}", data=u)
    
    # h5file.create_dataset("time", data=conc_time)
    # h5file.create_dataset("conc", data=conc)
    # h5file.create_dataset("conc_std", data=conc_std)
    # h5file.close()
    
# plt.subplots(1, 2, figsize=(25,10))
# plt.subplots_adjust(wspace=0.05)
# plt.subplot(1,2,1)
# # plt.contourf(x, y, u.T, 100, cmap="Greens")
# plt.imshow(u.T, cmap="Greens", origin="lower")
# plt.xticks(np.linspace(0, nx-1, 5), np.round(np.linspace(*bounds_x, 5), 2))
# plt.yticks(np.linspace(0, ny-1, 5), np.round(np.linspace(*bounds_y, 5), 2))
# plt.colorbar(ticks=np.linspace(np.min(u), np.min(u)+0.9*(np.max(u)-np.min(u)), 7))
# # plt.contourf(x, y, vort_sign.T, 1, cmap="coolwarm", alpha=0.2, antialiased=True)
# # plt.scatter(*conv_scale(vortices[strengths>0]).T, c="b", s=20)
# # plt.scatter(*conv_scale(vortices[strengths<0]).T, c="r", s=20)
# # plt.scatter(*particles.T, c="k", s=10, alpha=0.3)
# # plt.quiver(x[::10], y[::10], vx_norm[::10,::10].T, vy_norm[::10,::10].T, m[::10,::10].T, cmap="hot", alpha=0.4)
# # plt.xlim(-0.5)
# plt.xlim(0, nx)
# plt.ylim(0, ny)
# # plt.xticks([])
# # plt.yticks([])
# plt.title(f"t = {t[n]:0.1f}")
# plt.subplot(1,2,2)
# plt.plot(conc_time, conc, c="g")
# # plt.fill_between(conc_time, np.array(conc)-np.array(conc_std), np.array(conc)+np.array(conc_std), color="g", alpha=0.3)
# plt.show()