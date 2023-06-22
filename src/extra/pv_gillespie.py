#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:12:36 2022

@author: valeriano
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm
import h5py
import os
import time

#%%

@njit
def g(p, Lx, Ly, r_bins, r_max, srep):
    hist = np.zeros(len(r_bins)-1)
    dr = r_bins[1]-r_bins[0]
    for i in range(len(p)-1):
        for j in range(i+1, len(p)):
            for lx in range(-srep, srep+1):
                for ly in range(-srep, srep+1):
                    dx = (p[j,0]-p[i,0])+lx*Lx
                    dy = (p[j,1]-p[i,1])+ly*Ly
                    
                    r = np.sqrt(dx**2+dy**2)
                    if r < r_max:
                        hist[int(r/dr)] += 2/(2*np.pi*r*dr)
            
    return hist/len(p)**2*Lx*Ly

@njit(parallel=True)
def g_rt(p, Lx, Ly, r_bins, r_max, nt, srep):
    t = np.linspace(0, 2*np.pi, nt)
    dt = t[1]-t[0]
    hist = np.zeros((len(r_bins)-1, nt-1))
    dr = r_bins[1]-r_bins[0]
    
    for i in prange(len(p)-1):
        for j in range(i+1, len(p)):
            for lx in range(-srep, srep+1):
                for ly in range(-srep, srep+1):
                    dx = (p[j,0]-p[i,0])+lx*Lx
                    dy = (p[j,1]-p[i,1])+ly*Ly
                    
                    r = np.sqrt(dx*dx+dy*dy)
                    theta = np.arctan2(dy,dx)%(2*np.pi)
                    
                    if 0 < r < r_max:
                        hist[int(r/dr),int(theta/dt)] += 1/(r*dt*dr)
                        hist[int(r/dr),int(((theta+np.pi)%(2*np.pi))/dt)] += 1/(r*dt*dr)
                        
    return hist/len(p)**2*Lx*Ly

@njit
def sign(x):
    if x > 0:
        return 1
    else:
        return -1

@njit
def dot2d(u, v):
    return u[0]*v[0]+u[1]*v[1]

@njit
def norm2d(x):
    return np.sqrt(x[0]*x[0]+x[1]*x[1])

@njit
def cross2d(u, v):
    return u[0]*v[1]-u[1]*v[0]

@njit
def theta_dot2d(u, v):
    return (np.arccos(dot2d(u, v) / (norm2d(u)*norm2d(v)))*sign(cross2d(u, v)))%(2*np.pi)

@njit(parallel=True)
def g_rt_vec(p, vec, Lx, Ly, r_bins, r_max, nt, srep):
    t = np.linspace(0, 2*np.pi, nt)
    dt = t[1]-t[0]
    hist = np.zeros((len(r_bins)-1, nt-1))
    dr = r_bins[1]-r_bins[0]
    
    d = np.zeros(2)
    
    for i in prange(len(p)):
        for j in range(len(p)):
            if i != j:
                for lx in range(-srep, srep+1):
                    for ly in range(-srep, srep+1):
                        d[0] = (p[j,0]-p[i,0])+lx*Lx
                        d[1] = (p[j,1]-p[i,1])+ly*Ly
                        
                        r = norm2d(d)
                        
                        if 0 < r < r_max and norm2d(vec[i]) > 0:
                            theta1 = theta_dot2d(vec[i], d)
                            hist[int(r/dr),int(theta1/dt)] += 1/(r*dt*dr)
                        
    return hist/len(p)**2*Lx*Ly


#%%

@njit
def distances(pos, Lx, Ly, R2):
    p_in_pairs = np.zeros(pos.shape[0], dtype=np.int32)
    for i in range(pos.shape[0]-1):
        for j in range(i+1, pos.shape[0]):
            dx = abs(pos[i,0]-pos[j,0])
            dy = abs(pos[i,1]-pos[j,1])
            if dx > Lx/2:
                dx = Lx - dx
            if dy > Ly/2:
                dy = Ly - dy
            if dx*dx+dy*dy <= R2:
                p_in_pairs[i] += 1
                p_in_pairs[j] += 1

    return p_in_pairs


@njit
def diffusion(pos, dt, D):
    return pos + np.random.normal(0, np.sqrt(2*D*dt), pos.shape)


@njit
def next_reaction_time(pos, b, d, a, R2, Lx, Ly):
    n = len(pos)

    p_in_pairs = np.zeros(n)

    rates = np.array([b*n, d*n, 0])

    for i in range(n-1):
        for j in range(i+1, n):
            dx = np.abs(pos[i,0]-pos[j,0])
            dy = np.abs(pos[i,1]-pos[j,1])
            if dx > Lx/2:
                dx = Lx - dx
            if dy > Ly/2:
                dy = Ly - dy
            if dx*dx+dy*dy <= R2:
                rates[2] += 1
                # p_in_pairs[i] += 1
                # p_in_pairs[j] += 1

    rates[2] *= a

    rates_sum = np.sum(rates)

    tau = np.random.exponential(1/rates_sum)
    
    return tau


@njit
def exec_reaction(pos, b, b_shift, d, a, R2, boundsx, boundsy):
    n = len(pos)
    
    Lx = boundsx[1] - boundsx[0]
    Ly = boundsy[1] - boundsy[0]
    
    p_in_pairs = np.zeros(n)
    # theta_pairs = [[]]*n

    rates = np.array([b*n, d*n, 0])

    for i in range(n-1):
        for j in range(i+1, n):
            dx = np.abs(pos[i,0]-pos[j,0])
            dy = np.abs(pos[i,1]-pos[j,1])
            if dx > Lx/2:
                dx = Lx - dx
            if dy > Ly/2:
                dy = Ly - dy
            if dx*dx+dy*dy <= R2:
                rates[2] += 1
                p_in_pairs[i] += 1
                p_in_pairs[j] += 1
                # theta_pairs[i].append(theta_dot2d())

    p_in_pairs /= np.sum(p_in_pairs)

    rates[2] *= a
    
    rates_sum = np.sum(rates)

    rates_cum = np.cumsum(rates/rates_sum)

    reaction = np.searchsorted(rates_cum, np.random.uniform(0,1))

    theta_it = 10

    if reaction == 0:
        new_part = (pos[np.random.randint(0,n)]+np.random.normal(0, b_shift, (1,2))
                    -np.array([boundsx[0], boundsy[0]]))%np.array([Lx,Ly])+np.array([boundsx[0], boundsy[0]])
        pos = np.vstack((pos, new_part))
    
    elif reaction == 1:
        idx = np.random.randint(0,n)
        pos = np.vstack((pos[:idx], pos[idx+1:]))

    else:
        idx = np.searchsorted(np.cumsum(p_in_pairs), np.random.uniform(0,1))
        pos = np.vstack((pos[:idx], pos[idx+1:]))
        # theta_it = theta_dot2d()

    return pos#, theta_it


#%%

@njit
def pv_field(x, y, pvs, strengths, boundsx, boundsy, periodic_repeats=2):
    Lx = boundsx[1]-boundsx[0]
    Ly = boundsy[1]-boundsy[0]
    
    vx = 0
    vy = 0
    
    for n in range(len(pvs)):
        vx_temp = 0
        vy_temp = 0
        
        for i in range(-periodic_repeats, periodic_repeats+1):
            dx = x-pvs[n,0]
            dy = y-pvs[n,1]
            
            if dx-i*Lx != 0 or dy != 0:
                vx_temp -= np.sin(dy)/(np.cosh(dx-i*Lx)-np.cos(dy))
            if dx != 0 or dy-i*Ly != 0:
                vy_temp += np.sin(dx)/(np.cosh(dy-i*Ly)-np.cos(dx))
            
        vx += strengths[n]/(4*np.pi)*vx_temp
        vy += strengths[n]/(4*np.pi)*vy_temp
        
    return vx, vy

@njit
def pv_field2(x, y, pvs, strengths, boundsx, boundsy, periodic_repeats=2):
    Lx = boundsx[1]-boundsx[0]
    Ly = boundsy[1]-boundsy[0]
    
    vx = 0
    vy = 0
    
    for n in range(len(pvs)):
        vx_temp = 0
        vy_temp = 0
        
        for i in range(-periodic_repeats, periodic_repeats+1):
            for j in range(-periodic_repeats, periodic_repeats+1):
                d2 = (x-pvs[n,0]-i*Lx)**2+(y-pvs[n,1]-j*Ly)**2
                if d2 != 0:
                    vx_temp -= (y-pvs[n,1]-j*Ly)/d2
                    vy_temp += (x-pvs[n,0]-i*Lx)/d2

        vx += strengths[n]/(2*np.pi)*vx_temp
        vy += strengths[n]/(2*np.pi)*vy_temp
        
    return vx, vy

@njit
def pv_field_rk23_step(pvs, dt, strengths, boundsx, boundsy, periodic_repeats):
    Lx = boundsx[1]-boundsx[0]
    Ly = boundsy[1]-boundsy[0]
    
    k1 = np.zeros_like(pvs)
    k2 = np.copy(k1)
    
    for n in range(len(pvs)):
        k1[n] = pv_field(pvs[n,0], pvs[n,1], pvs, strengths, boundsx, boundsy, periodic_repeats)
    for n in range(len(pvs)):
        k2[n] = pv_field(pvs[n,0]+k1[n,0]*dt/2, pvs[n,1]+k1[n,1]*dt/2, pvs+k1*dt/2, strengths, boundsx, boundsy, periodic_repeats)
        
    pvs += dt*k2
    
    pvs[:,0] = (pvs-np.array([boundsx[0], boundsy[0]]))%np.array([Lx, Ly]) + np.array(boundsx[0], boundsy[0])
    
    return pvs

@njit
def pv_field_particles_rk23_step(pvs, pos, dt, strengths, boundsx, boundsy, periodic_repeats, D=0, n_steps=1):
    
    k1 = np.zeros_like(pvs)
    k2 = np.zeros_like(pvs)
    
    k1_pos = np.zeros_like(pos)
    k2_pos = np.zeros_like(pos)
    
    for i in range(n_steps):
        Lx = boundsx[1]-boundsx[0]
        Ly = boundsy[1]-boundsy[0]
        
        for n in range(len(pvs)):
            k1[n] = pv_field(pvs[n,0], pvs[n,1], pvs, strengths, boundsx, boundsy, periodic_repeats)
        for n in range(len(pvs)):
            k2[n] = pv_field(pvs[n,0]+k1[n,0]*dt/2, pvs[n,1]+k1[n,1]*dt/2, pvs+k1*dt/2, strengths, 
                              boundsx, boundsy, periodic_repeats)
            
        for n in range(len(pos)):
            k1_pos[n] = pv_field(pos[n,0], pos[n,1], pvs, strengths, boundsx, boundsy, periodic_repeats)
        for n in range(len(pos)):
            k2_pos[n] = pv_field(pos[n,0]+k1_pos[n,0]*dt/2, pos[n,1]+k1_pos[n,1]*dt/2, pvs+k1*dt/2, 
                                  strengths, boundsx, boundsy, periodic_repeats)
            
        pvs += dt*k2
        pos += dt*k2_pos
        
        if D > 0:
            pos = diffusion(pos, dt, D)
        
        pvs = (pvs-np.array([boundsx[0], boundsy[0]]))%np.array([Lx, Ly]) + np.array([boundsx[0], boundsy[0]])
        pos = (pos-np.array([boundsx[0], boundsy[0]]))%np.array([Lx, Ly]) + np.array([boundsx[0], boundsy[0]])
    
    return pvs, pos#, k2_pos


#%%

@njit
def time_evol_step(pos, pvs, strengths, D, b, b_shift, d, a, R2, boundsx, boundsy, periodic_repeats, reac_dt, advec_dt):
    Lx = boundsx[1]-boundsx[0]
    Ly = boundsy[1]-boundsy[0]

    tau = next_reaction_time(pos, b, d, a, R2, Lx, Ly)
    reac_dt += tau
    
    if reac_dt > advec_dt:
        pvs, pos = pv_field_particles_rk23_step(pvs, pos, advec_dt, strengths, boundsx, 
                                                boundsy, periodic_repeats, D, int(reac_dt/advec_dt))[:2]
        
        reac_dt -= advec_dt*int(reac_dt/advec_dt)
    
    pos = (pos-np.array([boundsx[0], boundsy[0]]))%np.array([Lx, Ly]) + np.array([boundsx[0], boundsy[0]])
    
    pvs, pos, vel = pv_field_particles_rk23_step(pvs, pos, reac_dt, strengths, boundsx, 
                                            boundsy, periodic_repeats, D)
    
    pos = exec_reaction(pos, b, b_shift, d, a, R2, boundsx, boundsy)
    
    pos = (pos-np.array([boundsx[0], boundsy[0]]))%np.array([Lx, Ly]) + np.array([boundsx[0], boundsy[0]])
    
    return pvs, pos, vel, reac_dt, tau

@njit
def time_evol_step2(pos, D, b, b_shift, d, a, R2, boundsx, boundsy, reac_dt, diff_dt):
    Lx = boundsx[1]-boundsx[0]
    Ly = boundsy[1]-boundsy[0]
    
    tau = next_reaction_time(pos, b, d, a, R2, Lx, Ly)
    reac_dt += tau
    
    if reac_dt > diff_dt:
        pos = diffusion(pos, diff_dt*int(reac_dt/diff_dt), D)
        
        reac_dt -= diff_dt*int(reac_dt/diff_dt)
    
    # pos = diffusion(pos, reac_dt, D)
    
    pos = exec_reaction(pos, b, b_shift, d, a, R2, boundsx, boundsy)
    
    pos = (pos-np.array([boundsx[0], boundsy[0]]))%np.array([Lx, Ly]) + np.array([boundsx[0], boundsy[0]])
    
    return pos, reac_dt, tau

#%%

n_pv = 10
periodic_repeats = 2
strength = 1

D = 1e-5

b = 0.15
b_shift = 0
d = 0.09
a = 0.013
R2 = 25e-4

dt = 1e-2

# print((b-d)/(2*a))
# print((b-d)/a)
# print((a+b-d)/(2*a))
# print((a+b-d)/a)
# print("\n")
# print((b-d)/(2*a)/(np.pi*R2))
# print((b-d)/a/(np.pi*R2))
# print((a+b-d)/(2*a)/(np.pi*R2))
# print((a+b-d)/a/(np.pi*R2))

n_eq = int((b-d)/(a)/(np.pi*R2))
# n_eq = int((b-d)/(np.pi*R2))
print(f"Mean field equilibrium: {n_eq} particles")

n0 = n_eq

boundsx = np.array([0.,1.])
boundsy = np.array([0.,1.])

Lx = boundsx[1]-boundsx[0]
Ly = boundsy[1]-boundsy[0]

# t = [0]
# npart = [n0]

# path = f"abm_rd_results/b{b}d{d}a{a}R{np.sqrt(R2)}D{D}pv{n_pv}"
# try:
#     os.mkdir(path)
# except FileExistsError:
#     pass

# plt.subplots(1, 2, figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.scatter(*pos.T, c="k", s=1)
# plt.scatter(*pvs[strengths>0].T, c="r", s=20)
# plt.scatter(*pvs[strengths<0].T, c="b", s=20)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.subplot(1, 2, 2)
# plt.plot(t, npart, c="k", lw=1)
# plt.suptitle(f"T = {0:.3f}: {pos.shape[0]} particles")
# plt.show()


#%%

tim = time.time()

# r_bins = np.linspace(0, 0.05, 41)
# r_bin_mids = (r_bins[1:]+r_bins[:-1])/2
# r_max = r_bins[-1]

# ntheta = 41
# theta = np.linspace(0, 2*np.pi, ntheta)
# dtheta = theta[1]-theta[0]
# theta_bin_mids = (theta[1:]+theta[:-1])/2

# hist2d_mean = np.zeros((r_bin_mids.shape[0], theta_bin_mids.shape[0]))
# hist2d_std = np.copy(hist2d_mean)

# T = 0
n_steps = 10000
# save_step = 1
# burn = 0

# reac_dt = 0

# # f = h5py.File("pcf2d.h5", "w")

# np.random.seed(0)
# pvs = np.random.uniform(0, 1, (n_pv,2))
# pos = np.random.uniform(0, 1, (n0, 2))
# strengths = strength*np.array([1.]*int(n_pv/2)+[-1.]*(n_pv-int(n_pv/2)))
# np.random.shuffle(strengths)

# dt = 1e-2

# # seeds = np.arange(1)
seeds = [1]

# r_pv_scale = 1
# b *= r_pv_scale
# d *= r_pv_scale
# a *= r_pv_scale

# t = [0]
# npart = [n0]

for seed in seeds:
    np.random.seed(seed)
    pvs = np.random.uniform(0, 1, (n_pv,2))
    strengths = strength*np.array([1.]*int(n_pv/2)+[-1.]*(n_pv-int(n_pv/2)))
    pos = np.random.uniform(0, 1, (1000, 2))
    
    # for n in tqdm(range(burn)):
    #     pvs, pos, vel, reac_dt, tau = time_evol_step(pos, pvs, strengths, D, b, b_shift, d, a, R2, 
    #                                     boundsx, boundsy, periodic_repeats, reac_dt, dt)
    
    tim = time.time()
    
    for n in tqdm(range(1, n_steps+1)):
        pvs, pos = pv_field_particles_rk23_step(pvs, pos, dt, strengths, boundsx, boundsy, periodic_repeats)[:2]
        
        # pvs, pos, vel, reac_dt, tau = time_evol_step(pos, pvs, strengths, D, b, b_shift, d, a, R2, 
        #                                     boundsx, boundsy, periodic_repeats, reac_dt, dt)
        
        # pos, reac_dt, tau = time_evol_step2(pos, D, b, b_shift, d, a, R2, boundsx, boundsy, reac_dt, dt)
        
        # T += tau
    
        # if n % save_step == 0:
            
        #     # idx = int(n/save_step)
            
        #     # hist2d = g_rt_vec(pos, vel, Lx, Ly, r_bins, r_max, ntheta, 1)
            
        #     # hist2d_mean_past = np.copy(hist2d_mean)
        #     # hist2d_mean += (hist2d-hist2d_mean)/idx
        #     # hist2d_std = np.sqrt((hist2d_std+(hist2d-hist2d_mean_past)*(hist2d-hist2d_mean))/idx)
            
        #     print(f"T = {T:.3f}: {pos.shape[0]} particles")
            
        #     t.append(T)
        #     npart.append(pos.shape[0])
            
        #     # plt.subplots(1, 2, figsize=(10,5))
        #     plt.figure(figsize=(10,10))
        #     # plt.subplot(1, 2, 1)
        #     plt.scatter(*pos.T, c="k", s=1)
        #     plt.scatter(*pvs[strengths>0].T, c="r", s=20)
        #     plt.scatter(*pvs[strengths<0].T, c="b", s=20)
        #     plt.xlim(0,1)
        #     plt.ylim(0,1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     # plt.subplot(1, 2, 2)
        #     # plt.plot(t, npart, c="k", lw=1)
        #     # plt.suptitle(f"T = {T:.3f}: {pos.shape[0]} particles")
        #     plt.show()
            
    # f.create_dataset(f"pcf2d_mean_seed{seed}", data=hist2d_mean)
    # f.create_dataset(f"pcf2d_std_seed{seed}", data=hist2d_std)
    
#         if pos.shape[0] == 0:
#             t.append(T)
#             npart.append(0)
#             print(f"T = {T:.3f}: {pos.shape[0]} particles")
#             break

# hist2d_mean = hist2d_mean[1:]
# hist2d_std = hist2d_std[1:]

# plt.plot(r_bin_mids[1:], np.sum(hist2d_mean, axis=1)/(ntheta-1))
# plt.hlines(1, 0, 0.01, "gray", "--")
# plt.show()

# theta2 = np.concatenate((theta_bin_mids, theta_bin_mids[-1:]+dtheta))
# hist2d2 = np.concatenate((hist2d_mean, hist2d_mean[:,0:1]), axis=1)
# hist2d2_std = np.concatenate((hist2d_std, hist2d_std[:,0:1]), axis=1)

# plt.subplots(figsize=(10,10), subplot_kw=dict(projection='polar'))
# plt.contourf(theta2, r_bin_mids[1:], hist2d2, 20, cmap="hot")
# plt.colorbar()
# plt.show()

# plt.subplots(subplot_kw=dict(projection='polar'))
# plt.plot(theta2, np.sum(hist2d2, axis=0)/(ntheta-1))
# plt.fill_between(theta2, np.sum(hist2d2-hist2d2_std, axis=0)/(ntheta-1), np.sum(hist2d2+hist2d2_std, axis=0)/(ntheta-1), alpha=0.5)
# plt.ylim(0, 1.1*np.max(np.sum(hist2d2, axis=0)/(ntheta-1)))
# plt.show()

# # plt.subplots(subplot_kw=dict(projection='polar'))
# plt.plot(theta2, np.sum(hist2d2, axis=0)/(ntheta-1))
# plt.fill_between(theta2, np.sum(hist2d2-hist2d2_std, axis=0)/(ntheta-1), np.sum(hist2d2+hist2d2_std, axis=0)/(ntheta-1), alpha=0.5)
# # plt.ylim(0, 1.2)
# plt.show()

# f.close()

print(time.time()-tim)

#%%

# seeds = np.arange(3)
# r_pv_scales = 10**np.arange(0, 2.5, 0.5)
# # r_pv_scales = 10**np.arange(2.5, 4.5, 0.5)

# n0 = n_eq
# n_steps = 20000
# save_step = 20

# L = max(Lx, Ly)

# r_bins = np.linspace(0, 0.05, 41)    
# r_bin_mids = (r_bins[1:]+r_bins[:-1])/2
# r_max = r_bins[-1]

# ntheta = 41
# theta = np.linspace(0, 2*np.pi, ntheta)
# dtheta = theta[1]-theta[0]
# theta_bin_mids = (theta[1:]+theta[:-1])/2

# for seed in seeds:
#     path = f"abm_rd_results/b{b}d{d}a{a}R{np.sqrt(R2)}N{n0}"
#     try:
#         os.mkdir(path)
#     except FileExistsError:
#         pass
    
#     runname = f"abm_rd_seed{seed}"
#     filename = f"{runname}.h5"
#     fig_folder = f"{runname}_figs"
    
#     if filename in os.listdir(path):
#         continue
    
#     try:
#         os.mkdir(f"{path}/{fig_folder}")
#     except FileExistsError:
#         pass
    
#     print(filename)
    
#     hist2d_mean = np.zeros((r_bin_mids.shape[0], theta_bin_mids.shape[0]))
#     hist2d_std = np.copy(hist2d_mean)
    
#     np.random.seed(seed)
#     pvs = np.random.uniform(0, 1, (n_pv,2))
#     pos = np.random.uniform(0, 1, (n0, 2))
#     strengths = strength*np.array([1.]*int(n_pv/2)+[-1.]*(n_pv-int(n_pv/2)))
#     np.random.shuffle(strengths)
    
#     t = [0]
#     npart = [n0]
    
#     hist2d = g_rt(pos, Lx, Ly, r_bins, r_max, ntheta, 1)
    
#     hist2d_mean_past = np.copy(hist2d_mean)
#     hist2d_mean += (hist2d-hist2d_mean)
#     hist2d_std = np.sqrt((hist2d_std+(hist2d-hist2d_mean_past)*(hist2d-hist2d_mean)))
    
#     T = 0
    
#     reac_dt = 0
    
#     for n in tqdm(range(1, n_steps+1)):
#         pos, reac_dt, tau = time_evol_step2(pos, D, b, b_shift, d, a, R2, boundsx, boundsy, reac_dt, dt)
        
#         T += tau

#         if pos.shape[0] == 0:
#             t.append(T)
#             npart.append(0)
#             break

#         if n % save_step == 0:
#             t.append(T)
#             npart.append(pos.shape[0])

#             idx = int(n/save_step)+1
            
#             hist2d = g_rt(pos, Lx, Ly, r_bins, r_max, ntheta, 1)
            
#             hist2d_mean_past = np.copy(hist2d_mean)
#             hist2d_mean += (hist2d-hist2d_mean)/idx
#             hist2d_std = np.sqrt((hist2d_std+(hist2d-hist2d_mean_past)*(hist2d-hist2d_mean))/idx)
            
#             # plt.subplots(1, 2, figsize=(20,10))
            
#             # plt.subplot(1, 2, 1)
#             # plt.scatter(*pos.T, c="k", s=10)
#             # plt.xlabel("X", fontsize=16)
#             # plt.ylabel("Y", fontsize=16)
#             # plt.xticks(fontsize=12)
#             # plt.yticks(fontsize=12)
#             # plt.xlim(0, Lx)
#             # plt.ylim(0, Ly)
            
#             # plt.subplot(1, 2, 2)
#             # plt.plot(g_bin_mid, pcf, c="k", lw=3)
#             # plt.xticks(fontsize=12)
#             # plt.yticks(fontsize=12)
#             # plt.xlabel("Distance", fontsize=16)
#             # plt.ylabel("Pair Correlation Function", fontsize=16)
#             # plt.hlines(1, 0, L, color="gray", ls="--", lw=2)
#             # plt.xlim(0, L)
#             # plt.ylim(0)
            
#             # plt.suptitle(f"T={T:.6f}", fontsize=22, y=0.92)
#             # # plt.show()
#             # plt.savefig(f"{path}/{fig_folder}/dist_pcf_t={T:.6f}", format="png",
#             #             dpi=300, bbox_inches="tight")
            
#             # plt.subplots(1, 2, figsize=(20,10))
            
#             # plt.subplot(1, 2, 1)
#             # plt.scatter(*pos.T, c="k", s=10)
#             # plt.xlabel("X", fontsize=16)
#             # plt.ylabel("Y", fontsize=16)
#             # plt.xticks(fontsize=12)
#             # plt.yticks(fontsize=12)
#             # plt.xlim(0, Lx)
#             # plt.ylim(0, Ly)
            
#             # plt.subplot(1, 2, 2)
#             # plt.plot(t, npart, c="k", lw=3)
#             # plt.xticks(fontsize=12)
#             # plt.yticks(fontsize=12)
#             # plt.xlabel("Time", fontsize=16)
#             # plt.ylabel("Population", fontsize=16)
#             # plt.xlim(0)
            
#             # plt.suptitle(f"T={T:.6f}", fontsize=22, y=0.92)
#             # # plt.show()
#             # plt.savefig(f"{path}/{fig_folder}/dist_pcf_t={T:.6f}", format="png",
#             #             dpi=300, bbox_inches="tight")
            
#             # fig = plt.figure(figsize=(12,10))
#             # plt.scatter(*pos.T, c="k", s=10)
#             # plt.xlabel("X", fontsize=16)
#             # plt.ylabel("Y", fontsize=16)
#             # plt.xticks(fontsize=12)
#             # plt.yticks(fontsize=12)
#             # plt.xlim(0, Lx)
#             # plt.ylim(0, Ly)
#             # plt.title(f"T={n}", fontsize=20)
#             # # plt.show()
#             # plt.savefig(f"{path}/{fig_folder}/dist_t={n}", format="png",
#             #             dpi=300, bbox_inches="tight")
#             # plt.close(fig)
            
#             # fig = plt.figure(figsize=(12,10))
#             # plt.plot(t, npart, c="k", lw=3)
#             # plt.xticks(fontsize=12)
#             # plt.yticks(fontsize=12)
#             # plt.xlabel("Time", fontsize=16)
#             # plt.ylabel("Population", fontsize=16)
#             # plt.xlim(0)
#             # # plt.show()
#             # plt.savefig(f"{path}/{fig_folder}/pop_t={n}", format="png",
#             #             dpi=300, bbox_inches="tight")
#             # plt.close(fig)
            
#             # fig = plt.figure(figsize=(12,10))
#             # plt.plot(g_bin_mid, pcf, c="k", lw=3)
#             # plt.xticks(fontsize=12)
#             # plt.yticks(fontsize=12)
#             # plt.xlabel("Distance", fontsize=16)
#             # plt.ylabel("Pair Correlation Function", fontsize=16)
#             # plt.hlines(1, 0, L, color="gray", ls="--", lw=2)
#             # plt.xlim(0, L)
#             # plt.ylim(0)
#             # plt.title(f"T={n}", fontsize=20)
#             # # plt.show()
#             # plt.savefig(f"{path}/{fig_folder}/pcf_t={n}", format="png",
#             #             dpi=300, bbox_inches="tight")
#             # plt.close(fig)
        
#     f = h5py.File(f"{path}/{filename}", "w")

#     f.attrs["seed"] = seed
    
#     f.attrs["n_pv"] = n_pv
#     f.attrs["periodic_repeats"] = periodic_repeats
    
#     f.attrs["n0"] = n0
#     f.attrs["b"] = b
#     f.attrs["b_shift"] = b_shift
#     f.attrs["d"] = d
#     f.attrs["a"] = a
#     f.attrs["R2"] = R2
#     f.attrs["D"] = D
    
#     f.attrs["Da"] = np.inf
    
#     f.attrs["dt"] = dt
#     f.attrs["n_steps"] = n_steps
    
#     f.attrs["boundsx"] = boundsx
#     f.attrs["boundsy"] = boundsy
    
#     f.create_dataset("t", data=t)
#     f.create_dataset("npart", data=npart)
#     f.create_dataset("pcf_mean", data=hist2d_mean)
#     f.create_dataset("pcf_std", data=hist2d_std)
    
#     f.close()
            
# ##############################################################################

# for r_pv_scale in r_pv_scales:
#     a1 = a*r_pv_scale
#     b1 = b*r_pv_scale
#     d1 = d*r_pv_scale
#     D1 = D
    
#     n_eq = int((b1-d1)/a1/(np.pi*R2))
#     n0 = n_eq
    
#     for seed in seeds:
#         path = f"abm_rd_results/b{b}d{d}a{a}R{np.sqrt(R2)}r_pv_scale{r_pv_scale:.2f}D_pv_scale{1/D:.2f}N{n0}pv{n_pv}"
#         try:
#             os.mkdir(path)
#         except FileExistsError:
#             pass
        
#         runname = f"abm_rd_r_pv_scale{r_pv_scale:.2f}D_pv_scale{1/D:.2f}_seed{seed}"
#         filename = f"{runname}.h5"
#         fig_folder = f"{runname}_figs"
        
#         if filename in os.listdir(path):
#             continue
        
#         try:
#             os.mkdir(f"{path}/{fig_folder}")
#         except FileExistsError:
#             pass
        
#         print(filename)
        
#         hist2d_mean = np.zeros((r_bin_mids.shape[0], theta_bin_mids.shape[0]))
#         hist2d_std = np.copy(hist2d_mean)
        
#         np.random.seed(seed)
#         pvs = np.random.uniform(0, 1, (n_pv,2))
#         pos = np.random.uniform(0, 1, (n0, 2))
#         strengths = strength*np.array([1.]*int(n_pv/2)+[-1.]*(n_pv-int(n_pv/2)))
#         np.random.shuffle(strengths)
        
#         t = [0]
#         npart = [n0]
        
#         reac_dt = 0
#         vel = time_evol_step(pos, pvs, strengths, D1, b1, b_shift, d1, a1, R2, 
#                               boundsx, boundsy, periodic_repeats, reac_dt, dt)[2]
        
#         hist2d = g_rt_vec(pos, vel, Lx, Ly, r_bins, r_max, ntheta, 1)
        
#         hist2d_mean_past = np.copy(hist2d_mean)
#         hist2d_mean += (hist2d-hist2d_mean)
#         hist2d_std = np.sqrt((hist2d_std+(hist2d-hist2d_mean_past)*(hist2d-hist2d_mean)))
        
#         T = 0
        
#         reac_dt = 0
        
#         for n in tqdm(range(1, n_steps)):
#             pvs, pos, vel, reac_dt, tau = time_evol_step(pos, pvs, strengths, D1, b1, b_shift, d1, a1, R2, 
#                                                 boundsx, boundsy, periodic_repeats, reac_dt, dt)
            
#             T += tau

#             if n % save_step == 0:
#                 t.append(T)
#                 npart.append(pos.shape[0])
                
#                 idx = int(n/save_step)+1
                
#                 hist2d = g_rt_vec(pos, vel, Lx, Ly, r_bins, r_max, ntheta, 1)
                
#                 hist2d_mean_past = np.copy(hist2d_mean)
#                 hist2d_mean += (hist2d-hist2d_mean)/idx
#                 hist2d_std = np.sqrt((hist2d_std+(hist2d-hist2d_mean_past)*(hist2d-hist2d_mean))/idx)
                
#                 # plt.subplots(1, 2, figsize=(20,10))
                
#                 # plt.subplot(1, 2, 1)
#                 # plt.scatter(*pos.T, c="k", s=10)
#                 # plt.xlabel("X", fontsize=16)
#                 # plt.ylabel("Y", fontsize=16)
#                 # plt.xticks(fontsize=12)
#                 # plt.yticks(fontsize=12)
#                 # plt.xlim(0, Lx)
#                 # plt.ylim(0, Ly)
                
#                 # plt.subplot(1, 2, 2)
#                 # plt.plot(g_bin_mid, pcf, c="k", lw=3)
#                 # plt.xticks(fontsize=12)
#                 # plt.yticks(fontsize=12)
#                 # plt.xlabel("Distance", fontsize=16)
#                 # plt.ylabel("Pair Correlation Function", fontsize=16)
#                 # plt.hlines(1, 0, L, color="gray", ls="--", lw=2)
#                 # plt.xlim(0, L)
#                 # plt.ylim(0)
                
#                 # plt.suptitle(f"T={T:.6f}", fontsize=22, y=0.92)
#                 # # plt.show()
#                 # plt.savefig(f"{path}/{fig_folder}/dist_pcf_t={T:.6f}", format="png",
#                 #             dpi=300, bbox_inches="tight")
                
#                 # plt.subplots(1, 2, figsize=(20,10))
                
#                 # plt.subplot(1, 2, 1)
#                 # plt.scatter(*pos.T, c="k", s=10)
#                 # plt.xlabel("X", fontsize=16)
#                 # plt.ylabel("Y", fontsize=16)
#                 # plt.xticks(fontsize=12)
#                 # plt.yticks(fontsize=12)
#                 # plt.xlim(0, Lx)
#                 # plt.ylim(0, Ly)
                
#                 # plt.subplot(1, 2, 2)
#                 # plt.plot(t, npart, c="k", lw=3)
#                 # plt.xticks(fontsize=12)
#                 # plt.yticks(fontsize=12)
#                 # plt.xlabel("Time", fontsize=16)
#                 # plt.ylabel("Population", fontsize=16)
#                 # plt.xlim(0)
                
#                 # plt.suptitle(f"T={T:.6f}", fontsize=22, y=0.92)
#                 # plt.show()
#                 # plt.savefig(f"{path}/{fig_folder}/dist_pcf_t={T:.6f}", format="png",
#                 #             dpi=300, bbox_inches="tight")
                
#                 # fig = plt.figure(figsize=(12,10))
#                 # plt.scatter(*pos.T, c="k", s=10)
#                 # plt.xlabel("X", fontsize=16)
#                 # plt.ylabel("Y", fontsize=16)
#                 # plt.xticks(fontsize=12)
#                 # plt.yticks(fontsize=12)
#                 # plt.xlim(0, Lx)
#                 # plt.ylim(0, Ly)
#                 # plt.title(f"T={n}", fontsize=20)
#                 # # plt.show()
#                 # plt.savefig(f"{path}/{fig_folder}/dist_t={n}", format="png",
#                 #             dpi=300, bbox_inches="tight")
#                 # plt.close(fig)
                
#                 # plt.figure(figsize=(12,10))
#                 # plt.plot(t, npart, c="k", lw=3)
#                 # plt.xticks(fontsize=12)
#                 # plt.yticks(fontsize=12)
#                 # plt.xlabel("Time", fontsize=16)
#                 # plt.ylabel("Population", fontsize=16)
#                 # plt.xlim(0)
#                 # # plt.show()
#                 # plt.savefig(f"{path}/{fig_folder}/pop_t={n}", format="png",
#                 #             dpi=300, bbox_inches="tight")
#                 # plt.close()
                
#                 # plt.figure(figsize=(12,10))
#                 # plt.plot(g_bin_mid, pcf, c="k", lw=3)
#                 # plt.xticks(fontsize=12)
#                 # plt.yticks(fontsize=12)
#                 # plt.xlabel("Distance", fontsize=16)
#                 # plt.ylabel("Pair Correlation Function", fontsize=16)
#                 # plt.hlines(1, 0, L, color="gray", ls="--", lw=2)
#                 # plt.xlim(0, L)
#                 # plt.ylim(0)
#                 # plt.title(f"T={n}", fontsize=20)
#                 # # plt.show()
#                 # plt.savefig(f"{path}/{fig_folder}/pcf_t={n}", format="png",
#                 #             dpi=300, bbox_inches="tight")
#                 # plt.close()
                
#             if len(pos) == 0:
#                 break
            
#         f = h5py.File(f"{path}/{filename}", "w")
        
#         f.attrs["r_pv_scale"] = r_pv_scale
#         f.attrs["seed"] = seed
        
#         f.attrs["n_pv"] = n_pv
#         f.attrs["periodic_repeats"] = periodic_repeats
        
#         f.attrs["n0"] = n0
#         f.attrs["b"] = b
#         f.attrs["b_shift"] = b_shift
#         f.attrs["d"] = d
#         f.attrs["a"] = a
#         f.attrs["R2"] = R2
#         f.attrs["D"] = D1
        
#         f.attrs["Da"] = d1/n_pv
        
#         f.attrs["dt"] = dt
#         f.attrs["n_steps"] = n_steps
        
#         f.attrs["boundsx"] = boundsx
#         f.attrs["boundsy"] = boundsy
        
#         f.create_dataset("t", data=t)
#         f.create_dataset("npart", data=npart)
#         f.create_dataset("pcf_mean", data=hist2d_mean)
#         f.create_dataset("pcf_std", data=hist2d_std)
        
#         f.close()