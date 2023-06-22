#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:49:36 2022

@author: valeriano
"""


#%%

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
import h5py
import os
import time
# from mpi4py import MPI
import sys

# comm = MPI.COMM_WORLD
# rank = comm.rank
# size = comm.size

#%%

@njit
def pcf_r(p, Lx, Ly, r_bins, r_max, srep):
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

@njit
def pcf_rt(p, Lx, Ly, r_bins, r_max, nt, srep):
    t = np.linspace(0, 2*np.pi, nt)
    dt = t[1]-t[0]
    hist = np.zeros((len(r_bins)-1, nt-1))
    dr = r_bins[1]-r_bins[0]
    
    for i in range(len(p)-1):
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

@njit
def g_rt_vec(p, vec, Lx, Ly, r_bins, r_max, nt, srep):
    t = np.linspace(0, 2*np.pi, nt)
    dt = t[1]-t[0]
    hist = np.zeros((len(r_bins)-1, nt-1))
    dr = r_bins[1]-r_bins[0]
    
    d = np.zeros(2)
    
    for i in range(len(p)):
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
def cell_nb(i, j, n):
    i = int(i)
    j = int(j)
    
    if n == 1:
        return np.array([[i,j]])
    
    else:
        return np.array([[i,j],
                         [(i+1)%n,j],
                         [(i-1)%n,j],
                         [i,(j+1)%n],
                         [i,(j-1)%n],
                         [(i-1)%n,(j-1)%n],
                         [(i-1)%n,(j+1)%n],
                         [(i+1)%n,(j+1)%n],
                         [(i+1)%n,(j-1)%n]])


@njit
def reorder_pos(pos, dx, bounds):
    divs = int((bounds[1]-bounds[0])/dx)
    d = (bounds[1]-bounds[0])/divs
    idxs = np.floor((pos-bounds[0])/d)
    reorder = np.argsort(divs*idxs[:,0] + idxs[:,1])
    new_pos = pos[reorder]
    idxs = idxs[reorder]    
    
    count_div = np.zeros(divs*divs, dtype=np.int64)
    compare = idxs[0]

    c = int(divs*idxs[0][0]+idxs[0][1])
    for i in range(idxs.shape[0]):
        # int(idxs[i])
        if (idxs[i] == compare).all():
            count_div[c] += 1
            
        elif c < len(count_div)-1:
            c = int(divs*idxs[i][0]+idxs[i][1])
            # print(idxs[i], c)
            compare = idxs[i]
            count_div[c] += 1
            
    idx_div = np.concatenate((np.array([0]), np.cumsum(count_div)))
     
    return new_pos, idx_div


@njit
def neighbors(pos_P, pos_NP, r, bounds):
    L = bounds[1]-bounds[0]
    
    pos_P, idx_div_P = reorder_pos(pos_P, r, bounds)
    pos_NP, idx_div_NP = reorder_pos(pos_NP, r, bounds)
    
    divs = int(L/r)
    d = L/divs
    
    nbs_P = np.zeros(pos_P.shape[0])
    nbs_NP = np.zeros(pos_NP.shape[0])

    r2 = r*r

    for i in range(0, np.where(idx_div_P==len(pos_P))[0][0]):
        idx = np.floor((pos_P[idx_div_P[i]]-bounds[0])/d)
        cell_nbs = cell_nb(idx[0], idx[1], divs)
        # print(idx)
        for j in range(idx_div_P[i], idx_div_P[i+1]):
            for cell in range(len(cell_nbs)):
                nbi = cell_nbs[cell][0]
                nbj = cell_nbs[cell][1]
                # print(idx, nbi, nbj, divs*nbi+nbj)
                for k in range(idx_div_P[divs*nbi+nbj], idx_div_P[divs*nbi+nbj+1]):
                    # print(nbi, nbj, j, k)
                    dx, dy = np.abs(pos_P[j]-pos_P[k])
                    if dx > L/2:
                        dx = L - dx
                    if dy > L/2:
                        dy = L - dy
                    if 0 < dx*dx+dy*dy <= r2:
                        nbs_P[j] += 1

                for k in range(idx_div_NP[divs*nbi+nbj], idx_div_NP[divs*nbi+nbj+1]):
                    dx, dy = np.abs(pos_P[j]-pos_NP[k])
                    if dx > L/2:
                        dx = L - dx
                    if dy > L/2:
                        dy = L - dy
                    if 0 < dx*dx+dy*dy <= r2:
                        nbs_P[j] += 1
                        
    for i in range(0, np.where(idx_div_NP==len(pos_NP))[0][0]):
        idx = np.floor((pos_NP[idx_div_NP[i]]-bounds[0])/d)
        cell_nbs = cell_nb(idx[0], idx[1], divs)
        # print(idx)
        for j in range(idx_div_NP[i], idx_div_NP[i+1]):
            for cell in range(len(cell_nbs)):
                nbi = cell_nbs[cell][0]
                nbj = cell_nbs[cell][1]
                # print(idx, nbi, nbj)
                for k in range(idx_div_P[divs*nbi+nbj], idx_div_P[divs*nbi+nbj+1]):
                    # print(nbi, nbj, j, k)
                    dx, dy = np.abs(pos_NP[j]-pos_P[k])
                    if dx > L/2:
                        dx = L - dx
                    if dy > L/2:
                        dy = L - dy
                    if 0 < dx*dx+dy*dy <= r2:
                        nbs_NP[j] += 1

                for k in range(idx_div_NP[divs*nbi+nbj], idx_div_NP[divs*nbi+nbj+1]):
                    dx, dy = np.abs(pos_NP[j]-pos_NP[k])
                    if dx > L/2:
                        dx = L - dx
                    if dy > L/2:
                        dy = L - dy
                    if 0 < dx*dx+dy*dy <= r2:
                        nbs_NP[j] += 1
     
    return pos_P, pos_NP, nbs_P, nbs_NP


@njit
def reorder_pos_nbs(pos, nbs, dx, bounds):
    divs = int((bounds[1]-bounds[0])/dx)
    d = (bounds[1]-bounds[0])/divs
    idxs = np.floor((pos-bounds[0])/d)
    reorder = np.argsort(divs*idxs[:,0] + idxs[:,1])
    new_pos = pos[reorder]
    new_nbs = nbs[reorder]
    idxs = idxs[reorder]    
    
    count_div = np.zeros(divs*divs, dtype=np.int64)
    compare = idxs[0]

    c = int(divs*idxs[0][0]+idxs[0][1])
    for i in range(idxs.shape[0]):
        # int(idxs[i])
        if (idxs[i] == compare).all():
            count_div[c] += 1
            
        elif c < len(count_div)-1:
            c = int(divs*idxs[i][0]+idxs[i][1])
            # print(idxs[i], c)
            compare = idxs[i]
            count_div[c] += 1
            
    idx_div = np.concatenate((np.array([0]), np.cumsum(count_div)))
     
    return new_pos, new_nbs, idx_div


@njit
def P_NP_fitness(pos_P, pos_NP, g, r, rcomp, bounds):
    L = bounds[1]-bounds[0]
    
    pos_P, pos_NP, nbs_P, nbs_NP = neighbors(pos_P, pos_NP, rcomp, bounds)
    
    pos_P, nbs_P, idx_div_P = reorder_pos_nbs(pos_P, nbs_P, r, bounds)
    pos_NP, nbs_NP, idx_div_NP = reorder_pos_nbs(pos_NP, nbs_NP, r, bounds)
    
    divs = int(L/r)
    
    d = L/divs
    
    fitness_P = np.zeros(pos_P.shape[0])
    fitness_NP = np.zeros(pos_NP.shape[0])

    r2 = r*r
    
    for i in range(0, np.where(idx_div_P==len(pos_P))[0][0]):
        idx = np.floor((pos_P[idx_div_P[i]]-bounds[0])/d)
        cell_nbs = cell_nb(idx[0], idx[1], divs)
        # print(idx)
        for j in range(idx_div_P[i], idx_div_P[i+1]):
            nbhd_P = []
            nbhd_NP = []
            for cell in range(len(cell_nbs)):
                nbi = cell_nbs[cell][0]
                nbj = cell_nbs[cell][1]
                # print(idx, nbi, nbj)
                for k in range(idx_div_P[divs*nbi+nbj], idx_div_P[divs*nbi+nbj+1]):

                    dx, dy = np.abs(pos_P[j]-pos_P[k])
                    if dx > L/2:
                        dx = L - dx
                    if dy > L/2:
                        dy = L - dy
                    if dx*dx+dy*dy <= r2:
                        nbhd_P.append(k)

                for k in range(idx_div_NP[divs*nbi+nbj], idx_div_NP[divs*nbi+nbj+1]):
                    dx, dy = np.abs(pos_P[j]-pos_NP[k])
                    if dx > L/2:
                        dx = L - dx
                    if dy > L/2:
                        dy = L - dy
                    if 0 < dx*dx+dy*dy <= r2:
                        nbhd_NP.append(k)
                        
            # print(j, "nbhd_P", nbhd_P)
            # print(j, "nbhd_NP", nbhd_NP)
            
            benefit = g#/(len(nbhd_P)+len(nbhd_NP))
            for nb in nbhd_P:
                fitness_P[nb] += benefit
            for nb in nbhd_NP:
                fitness_NP[nb] += benefit
                        
    return pos_P, pos_NP, fitness_P, fitness_NP, nbs_P, nbs_NP


@njit
def diffusion(pos, dt, D):
    return pos + np.random.normal(0, np.sqrt(2*D*dt), pos.shape)


@njit
def next_reaction_time(pos_P, pos_NP, b, d, k, g, a, r, rcomp, bounds):
    n_P = len(pos_P)
    n_NP = len(pos_NP)
    
    pos_P, pos_NP, fitness_P, fitness_NP, nbs_P, nbs_NP = P_NP_fitness(pos_P, pos_NP, g, r, rcomp, bounds)

    rates = np.array([d*(n_P+n_NP), (b-k)*n_P+fitness_P.sum(), b*n_NP+fitness_NP.sum(), 
                      a*nbs_P.sum(), a*nbs_NP.sum()])

    rates_sum = np.sum(rates)

    tau = np.random.exponential(1/rates_sum)
    
    return tau


@njit
def exec_reaction(pos_P, pos_NP, b, b_shift, d, k, g, a, r, rcomp, bounds, force_reac=-1):
    n_P = len(pos_P)
    n_NP = len(pos_NP)
    L = bounds[1]-bounds[0]
    
    pos_P, pos_NP, fitness_P, fitness_NP, nbs_P, nbs_NP = P_NP_fitness(pos_P, pos_NP, g, r, rcomp, bounds)

    rates = np.array([d*(n_P+n_NP), (b-k)*n_P+fitness_P.sum(), b*n_NP+fitness_NP.sum(), 
                      a*nbs_P.sum(), a*nbs_NP.sum()])
    
    rates_sum = np.sum(rates)
    rates_cum = np.cumsum(rates/rates_sum)

    if force_reac == -1:
        reaction = np.searchsorted(rates_cum, np.random.uniform(0,1))
    else:
        reaction = force_reac

    if reaction == 0:
        idx = np.random.randint(0,n_P+n_NP)
        if idx < n_P:
            pos_P = np.vstack((pos_P[:idx], pos_P[idx+1:]))
        else:
            idx = np.random.randint(0,n_NP)
            pos_NP = np.vstack((pos_NP[:idx], pos_NP[idx+1:]))

    elif reaction == 1:
        birth_P_cum = np.cumsum(b-k+fitness_P)/((b-k)*n_P+fitness_P.sum())
        new_part = (pos_P[np.searchsorted(birth_P_cum, np.random.uniform(0,1))]+np.random.normal(0, b_shift, (1,2))
                    -bounds[0])%L+bounds[0]
        pos_P = np.vstack((pos_P, new_part))

    elif reaction == 2:
        birth_NP_cum = np.cumsum(b+fitness_NP)/(b*n_NP+fitness_NP.sum())
        new_part = (pos_NP[np.searchsorted(birth_NP_cum, np.random.uniform(0,1))]+np.random.normal(0, b_shift, (1,2))
                    -bounds[0])%L+bounds[0]
        pos_NP = np.vstack((pos_NP, new_part))
        
    elif reaction == 3:
        comp_P_cum = np.cumsum(nbs_P/nbs_P.sum())
        idx = np.searchsorted(comp_P_cum, np.random.random())
        pos_P = np.vstack((pos_P[:idx], pos_P[idx+1:]))
    
    elif reaction == 4:
        comp_NP_cum = np.cumsum(nbs_NP/nbs_NP.sum())
        idx = np.searchsorted(comp_NP_cum, np.random.random())
        pos_NP = np.vstack((pos_NP[:idx], pos_NP[idx+1:]))

    return pos_P, pos_NP


#%%

@njit
def ch_flow(y, a, b, s):
    v = np.zeros((len(y), 2))
    v[:,0] = s*(y-a)*(y-b)*4/((a-b)*(a-b))
    
    return v


@njit
def pv_field(x, y, pvs, strengths, bounds, periodic_repeats=2):
    L = bounds[1]-bounds[0]
    
    vx = 0
    vy = 0
    
    for n in range(len(pvs)):
        vx_temp = 0
        vy_temp = 0
        
        for i in range(-periodic_repeats, periodic_repeats+1):
            dx = x-pvs[n,0]
            dy = y-pvs[n,1]
            
            if dx-i*L != 0 or dy != 0:
                vx_temp -= np.sin(2*np.pi*dy/L)/(np.cosh(2*np.pi*dx/L-2*np.pi*i)-np.cos(2*np.pi*dy/L))
            if dx != 0 or dy-i*L != 0:
                vy_temp += np.sin(2*np.pi*dx/L)/(np.cosh(2*np.pi*dy/L-2*np.pi*i)-np.cos(2*np.pi*dx/L))
            
        vx += strengths[n]/(2*L)*vx_temp
        vy += strengths[n]/(2*L)*vy_temp
        
    return vx, vy


# @njit
# def pv_field(x, y, pvs, strengths, bounds, periodic_repeats=2):
#     L = bounds[1]-bounds[0]
    
#     vx = 0
#     vy = 0
    
#     for n in range(len(pvs)):
#         vx_temp = 0
#         vy_temp = 0
        
#         for i in range(-periodic_repeats, periodic_repeats+1):
#             for j in range(-periodic_repeats, periodic_repeats+1):
#                 d2 = (x-pvs[n,0]-i*L)**2+(y-pvs[n,1]-j*L)**2
#                 if d2 != 0:
#                     vx_temp -= (y-pvs[n,1]-j*L)/d2
#                     vy_temp += (x-pvs[n,0]-i*L)/d2

#         vx += strengths[n]/(2*np.pi)*vx_temp
#         vy += strengths[n]/(2*np.pi)*vy_temp
        
#     return vx, vy

@njit
def pv_field_particles(pvs, pos_P, pos_NP, strengths, bounds, periodic_repeats):
    v_pos_P = np.zeros_like(pos_P)
    v_pos_NP = np.zeros_like(pos_NP)
        
    for n in range(len(pos_P)):
        v_pos_P[n] = pv_field(pos_P[n,0], pos_P[n,1], pvs, strengths, bounds, periodic_repeats)

    for n in range(len(pos_NP)):
        v_pos_NP[n] = pv_field(pos_NP[n,0], pos_NP[n,1], pvs, strengths, bounds, periodic_repeats)
    
    return v_pos_P, v_pos_NP


@njit
def pv_field_particles_rk23_step(pvs, pos_P, pos_NP, dt, strengths, bounds, periodic_repeats, D=0, n_steps=1):
    L = bounds[1]-bounds[0]
    
    k1 = np.zeros_like(pvs)
    # k2 = np.zeros_like(pvs)
    
    k1_pos_P = np.zeros_like(pos_P)
    # k2_pos_P = np.zeros_like(pos_P)
    
    k1_pos_NP = np.zeros_like(pos_NP)
    # k2_pos_NP = np.zeros_like(pos_NP)
    
    for i in range(n_steps):
        
        for n in range(len(pvs)):
            k1[n] = pv_field(pvs[n,0], pvs[n,1], pvs, strengths, bounds, periodic_repeats)
        # for n in range(len(pvs)):
        #     k2[n] = pv_field(pvs[n,0]+k1[n,0]*dt/2, pvs[n,1]+k1[n,1]*dt/2, pvs+k1*dt/2, strengths, 
        #                       bounds, periodic_repeats)
            
        for n in range(len(pos_P)):
            k1_pos_P[n] = pv_field(pos_P[n,0], pos_P[n,1], pvs, strengths, bounds, periodic_repeats)
        # for n in range(len(pos_P)):
        #     k2_pos_P[n] = pv_field(pos_P[n,0]+k1_pos_P[n,0]*dt/2, pos_P[n,1]+k1_pos_P[n,1]*dt/2, pvs+k1*dt/2, 
        #                           strengths, bounds, periodic_repeats)
            
        for n in range(len(pos_NP)):
            k1_pos_NP[n] = pv_field(pos_NP[n,0], pos_NP[n,1], pvs, strengths, bounds, periodic_repeats)
        # for n in range(len(pos_NP)):
        #     k2_pos_NP[n] = pv_field(pos_NP[n,0]+k1_pos_NP[n,0]*dt/2, pos_NP[n,1]+k1_pos_NP[n,1]*dt/2, pvs+k1*dt/2, 
        #                           strengths, bounds, periodic_repeats)
            
        pvs += dt*k1
        pos_P += dt*k1_pos_P
        pos_NP += dt*k1_pos_NP
        
        if D > 0:
            pos_P = diffusion(pos_P, dt, D)
            pos_NP = diffusion(pos_NP, dt, D)
        
        pvs = (pvs-bounds[0])%L + bounds[0]
        pos_P = (pos_P-bounds[0])%L + bounds[0]
        pos_NP = (pos_NP-bounds[0])%L + bounds[0]
    
    return pvs, pos_P, pos_NP, k1_pos_P, k1_pos_NP


@njit
def field_particles_rk23_step(pos_P, pos_NP, dt, strength, bounds, periodic_repeats, D=0, n_steps=1):
    L = bounds[1]-bounds[0]
    
    for i in range(n_steps):
        
        k1_pos_P = ch_flow(pos_P[:,1], bounds[0], bounds[1], strength)
        k2_pos_P = ch_flow(pos_P[:,1]+k1_pos_P[:,1]*dt/2, bounds[0], bounds[1], strengths[0])
        
        k1_pos_NP = ch_flow(pos_NP[:,1], bounds[0], bounds[1], strength)
        k2_pos_NP = ch_flow(pos_NP[:,1]+k1_pos_NP[:,1]*dt/2, bounds[0], bounds[1], strengths[0])
    
        pos_P += dt*k2_pos_P
        pos_NP += dt*k2_pos_NP
        
        if D > 0:
            pos_P = diffusion(pos_P, dt, D)
            pos_NP = diffusion(pos_NP, dt, D)
        
        pos_P = (pos_P-bounds[0])%L + bounds[0]
        pos_NP = (pos_NP-bounds[0])%L + bounds[0]
    
    return pos_P, pos_NP, k2_pos_P, k2_pos_NP


@njit
def time_evol_step(pos_P, pos_NP, pvs, strengths, D, b, b_shift, d, k, g, a, r, rcomp, bounds, periodic_repeats, reac_dt, advec_dt, force_reac=-1):
    L = bounds[1]-bounds[0]

    tau = next_reaction_time(pos_P, pos_NP, b, d, k, g, a, r, rcomp, bounds)
    reac_dt += tau
    
    if reac_dt > advec_dt:
        pvs, pos_P, pos_NP = pv_field_particles_rk23_step(pvs, pos_P, pos_NP, advec_dt, strengths, bounds, 
                                                          periodic_repeats, D, int(reac_dt/advec_dt))[:3]
        
        reac_dt -= advec_dt*int(reac_dt/advec_dt)

    pos_P, pos_NP = exec_reaction(pos_P, pos_NP, b, b_shift, d, k, g, a, r, rcomp, bounds, force_reac)

    pvs, pos_P, pos_NP, v_pos_P, v_pos_NP = pv_field_particles_rk23_step(pvs, pos_P, pos_NP, reac_dt, strengths, 
                                                    bounds, periodic_repeats, D)
    reac_dt = 0
        
    return pvs, pos_P, pos_NP, reac_dt, tau


@njit
def time_evol_step2(pos_P, pos_NP, D, b, b_shift, d, k, g, a, r, rcomp, bounds, reac_dt, diff_dt):
    L = bounds[1]-bounds[0]
    
    tau = next_reaction_time(pos_P, pos_NP, b, d, k, g, a, r, rcomp, bounds)
    reac_dt += tau
    
    if reac_dt > diff_dt:
        pos_P = diffusion(pos_P, diff_dt*int(reac_dt/diff_dt), D)
        pos_NP = diffusion(pos_NP, diff_dt*int(reac_dt/diff_dt), D)
        
        pos_P = (pos_P-bounds[0])%L + bounds[0]
        pos_NP = (pos_NP-bounds[0])%L + bounds[0]
        
        reac_dt -= diff_dt*int(reac_dt/diff_dt)
        
    pos_P, pos_NP = exec_reaction(pos_P, pos_NP, b, b_shift, d, k, g, a, r, rcomp, bounds)
    
    pos_P = diffusion(pos_P, reac_dt, D)
    pos_NP = diffusion(pos_NP, reac_dt, D)

    pos_P = (pos_P-bounds[0])%L + bounds[0]
    pos_NP = (pos_NP-bounds[0])%L + bounds[0]

    reac_dt = 0
    
    return pos_P, pos_NP, reac_dt, tau


@njit
def time_evol_step3(pos_P, pos_NP, strength, D, b, b_shift, d, k, g, r, bounds, periodic_repeats, reac_dt, advec_dt, force_reac=-1):
    L = bounds[1]-bounds[0]

    tau = next_reaction_time(pos_P, pos_NP, b, d, k, g, r, bounds)
    reac_dt += tau
    
    if reac_dt > advec_dt:
        pos_P, pos_NP = field_particles_rk23_step(pos_P, pos_NP, advec_dt, strength, bounds, periodic_repeats, D, int(reac_dt/advec_dt))[:2]
        
        reac_dt -= advec_dt*int(reac_dt/advec_dt)
    
    pos_P = (pos_P-bounds[0])%L + bounds[0]
    pos_NP = (pos_NP-bounds[0])%L + bounds[0]
    
    pos_P, _NP = field_particles_rk23_step(pos_P, pos_NP, reac_dt, strength, 
                                                   bounds, periodic_repeats, D)[:2]
    
    pos_P = (pos_P-bounds[0])%L + bounds[0]
    pos_NP = (pos_NP-bounds[0])%L + bounds[0]
    
    pos_P, pos_NP = exec_reaction(pos_P, pos_NP, b, b_shift, d, k, g, r, bounds, force_reac)
    
    pos_P, _NP, v_pos_P, v_pos_NP = field_particles_rk23_step(pos_P, pos_NP, 0, strength, 
                                                   bounds, periodic_repeats, D)
    
    # pos_P = (pos_P-bounds[0])%L + bounds[0]
    # pos_NP = (pos_NP-bounds[0])%L + bounds[0]
    
    pos_P[:,0] = (pos_P[:,0]-bounds[0])%L + bounds[0]
    pos_NP[:,0] = (pos_NP[:,0]-bounds[0])%L + bounds[0]
    
    pos_P[pos_P[:,1]<bounds[0]] = bounds[0]
    pos_P[pos_P[:,1]>bounds[1]] = bounds[1]
    
    pos_NP[pos_NP[:,1]<bounds[0]] = bounds[0]
    pos_NP[pos_NP[:,1]>bounds[1]] = bounds[1]
    
    return pos_P, pos_NP, v_pos_P, v_pos_NP, reac_dt, tau


@njit
def set_seed(value):
    np.random.seed(value)
    
#%%

n_pv = 5
periodic_repeats = 2

D = 1e-5

b = 0.8
b_shift = 0
d = 1.
g = 0.7
k = 0.1
r = 0.05
rcomp = 0.15

n0_P = 100
n0_NP = 0

strength = 0.
seed = 1

# b, g, k, r, strength, seed = np.array(sys.argv[1:], dtype=np.float64)

a = (b-d-k+g)/10

# a = 0.025789

seed = int(seed)

bounds = np.array([0.,1.])

L = bounds[1]-bounds[0]

n_steps = int(1e10)
save_step = 1000

path = "abm_pg_pv_comp"
try:
    os.mkdir(path)
except FileExistsError:
    pass


#%%

T = 0
reac_dt = 0

t = [0]
npart_P = [n0_P]
npart_NP = [n0_NP]

np.random.seed(seed)
set_seed(seed)
pvs = np.random.uniform(0, 2*np.pi, (n_pv,2))
strengths = strength*np.array([1.]*int(n_pv/2)+[-1.]*(n_pv-int(n_pv/2)))
np.random.shuffle(strengths)
pos_P = np.random.uniform(0, 1, (n0_P, 2))
pos_NP = np.random.uniform(0, 1, (n0_NP, 2))

if strength == 0:
    dt = 1e-3/(2*D)
else:
    v_pos_P, v_pos_NP = pv_field_particles(pvs, pos_P, pos_NP, strengths, bounds, periodic_repeats)
    dt = 1e-2/np.mean(np.hypot(*np.concatenate((v_pos_P, v_pos_NP)).T))
    dt = 1e-2/(strength*n_pv**0.5/(2*np.pi))

runname = f"b{b}d{d}g{g}k{k}a{a:f}_r{r}_v{strength}_D{D}_seed{seed}"

print(runname)

# os.mkdir(f"{path}/{runname}")

# with open(f"{path}/{runname}/params.txt", "w") as params_file:
#     params_file.write(f"seed {seed}")
    
#     params_file.write(f"b {b}\n")
#     params_file.write(f"d {d}\n")
#     params_file.write(f"g {g}\n")
#     params_file.write(f"k {k}\n")
#     params_file.write(f"a {a}\n")
#     params_file.write(f"r {r}\n")
#     params_file.write(f"D {D}\n")
    
#     params_file.write(f"bounds {bounds[0]} {bounds[1]}\n")
#     params_file.write(f"L {L}\n")
    
#     params_file.write(f"dt {dt}\n")
    
#     params_file.write(f"n0_P {n0_P}\n")
#     params_file.write(f"n0_NP {n0_NP}\n")
    
#     params_file.write(f"n_pv {n_pv}\n")
#     params_file.write(f"strength {strength}\n")
#     params_file.write(f"periodic_repeats {periodic_repeats}\n")
    
#     params_file.write(f"n_steps {n_steps}\n")
#     params_file.write(f"save_step {save_step}\n")

# with open(f"{path}/{runname}/macro_pop.csv", "w") as macro_pop:
#     macro_pop.write(f"0.0,{n0_P},{n0_NP}\n")

# os.mkdir(f"{path}/{runname}/pos_P")
# os.mkdir(f"{path}/{runname}/pos_NP")
# os.mkdir(f"{path}/{runname}/v_pos_P")
# os.mkdir(f"{path}/{runname}/v_pos_NP")

# np.save(f"{path}/{runname}/pos_P/t0-0.0.npy", pos_P)
# np.save(f"{path}/{runname}/pos_NP/t0-0.0.npy", pos_NP)
# if strength != 0:
#     np.save(f"{path}/{runname}/v_pos_P/t0-0.0.npy", v_pos_P)
#     np.save(f"{path}/{runname}/v_pos_NP/t0-0.0.npy", v_pos_NP)

plt.subplots(1, 2, figsize=(10,5))
# plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.scatter(*pos_P.T, c="b", s=1)
plt.scatter(*pos_NP.T, c="r", s=1)
# plt.scatter(*pvs[strengths>0].T, c="r", s=20)
# plt.scatter(*pvs[strengths<0].T, c="b", s=20)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.plot(t, npart_P, c="b", lw=1)
plt.plot(t, npart_NP, c="r", lw=1)
# plt.suptitle(f"T = {T:.3f}: {pos.shape[0]} particles")
plt.show()

if strength == 0:
    for n in tqdm(range(1, n_steps+1)):
        # pvs, pos_P, pos_NP, reac_dt, tau = time_evol_step(pos_P, pos_NP, pvs, strengths, D, b, b_shift, d, k, g, r, 
        #                                     bounds, periodic_repeats, reac_dt, dt)
        
        pos_P, pos_NP, reac_dt, tau = time_evol_step2(pos_P, pos_NP, D, b, b_shift, d, k, g, 
                                                      a, r, rcomp, bounds, reac_dt, dt)

        T += tau
    
        if n % save_step == 0:

            # with open(f"{path}/{runname}/macro_pop.csv", "a") as macro_pop:
            #     macro_pop.write(f"{T},{pos_P.shape[0]},{pos_NP.shape[0]}\n")
            
            # np.save(f"{path}/{runname}/pos_P/t{int(n/save_step)}-{T}.npy", pos_P)
            # np.save(f"{path}/{runname}/pos_NP/t{int(n/save_step)}-{T}.npy", pos_NP)

            t.append(T)
            npart_P.append(pos_P.shape[0])
            npart_NP.append(pos_NP.shape[0])

            plt.subplots(1, 2, figsize=(10,5))
            # plt.figure(figsize=(10,10))
            plt.subplot(1, 2, 1)
            plt.scatter(*pos_P.T, c="b", s=1)
            plt.scatter(*pos_NP.T, c="r", s=1)
            # plt.scatter(*pvs[strengths>0].T, c="r", s=20)
            # plt.scatter(*pvs[strengths<0].T, c="b", s=20)
            plt.xlim(0,L)
            plt.ylim(0,L)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 2, 2)
            plt.plot(t, npart_P, c="b", lw=1)
            plt.plot(t, npart_NP, c="r", lw=1)
            # plt.suptitle(f"T = {T:.3f}: {pos.shape[0]} particles")
            plt.show()

        # if pos_P.shape[0] == 0 or pos_NP.shape[0] == 0:

        #     # with open(f"{path}/{runname}/macro_pop.csv", "a") as macro_pop:
        #     #     macro_pop.write(f"{T},{pos_P.shape[0]},{pos_NP.shape[0]}\n")
            
        #     # np.save(f"{path}/{runname}/pos_P/t{int(n/save_step)}-{T}.npy", pos_P)
        #     # np.save(f"{path}/{runname}/pos_NP/t{int(n/save_step)}-{T}.npy", pos_NP)
            
        #     break
else:
    # for n in tqdm(range(1)):
    #     pvs, pos_P, pos_NP, reac_dt, tau = time_evol_step(pos_P, pos_NP, pvs, 
    #                                                       strengths, D, b, b_shift, d, k, g, r, 
    #                                                       bounds, periodic_repeats, reac_dt, dt, 2)
        
        # pos_P, pos_NP, v_pos_P, v_pos_NP, reac_dt, tau = time_evol_step3(pos_P, pos_NP, strength, D, b, b_shift, 
        #                                                                   d, k, g, r, bounds, periodic_repeats, 
        #                                                                     reac_dt, dt, 2)
        
        # plt.subplots(1, 2, figsize=(10,5))
        # # plt.figure(figsize=(10,10))
        # plt.subplot(1, 2, 1)
        # plt.scatter(*pos_P.T, c="b", s=1)
        # plt.scatter(*pos_NP.T, c="r", s=1)
        # plt.scatter(*pvs[strengths>0].T, facecolors="w", edgecolors="k", s=20)
        # plt.scatter(*pvs[strengths<0].T, c="k", s=20)
        # plt.xlim(0,1)
        # plt.ylim(0,1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(1, 2, 2)
        # plt.plot(t, npart_P, c="b", lw=1)
        # plt.plot(t, npart_NP, c="r", lw=1)
        # # plt.suptitle(f"T = {T:.3f}: {pos.shape[0]} particles")
        # plt.show()
        
    for n in tqdm(range(1, n_steps+1)):
        # pvs, pos_P, pos_NP, v_pos_P, v_pos_NP = pv_field_particles_rk23_step(pvs, pos_P, pos_NP, dt,
        #                                                   strengths, bounds, periodic_repeats, D)
        
        pvs, pos_P, pos_NP, reac_dt, tau = time_evol_step(pos_P, pos_NP, pvs, 
                                                          strengths, D, b, b_shift, d, k, g, a, r, 
                                                          rcomp, bounds, periodic_repeats, reac_dt, dt)
        
        # pos_P, pos_NP, reac_dt, tau = time_evol_step2(pos_P, pos_NP, D, b, b_shift, d, k, g, r, bounds, reac_dt, dt)
        
        # pos_P, pos_NP, v_pos_P, v_pos_NP, reac_dt, tau = time_evol_step3(pos_P, pos_NP, strength, D, b, b_shift, 
        #                                                                   d, k, g, r, bounds, periodic_repeats, 
        #                                                                   reac_dt, dt)

        # T = n*dt
        T += tau
        
        v_pos_P, v_pos_NP = pv_field_particles(pvs, pos_P, pos_NP, strengths, bounds, periodic_repeats)
        
        dt = 1e-2/np.mean(np.hypot(*np.concatenate((v_pos_P, v_pos_NP)).T))
    
        if n % save_step == 0:
            
            v_pos_P, v_pos_NP = pv_field_particles(pvs, pos_P, pos_NP, strengths, bounds, periodic_repeats)
            
            dt = 1e-2/np.mean(np.hypot(*np.concatenate((v_pos_P, v_pos_NP)).T))
            # dt = 1e-2/(strength*n_pv**0.5/(2*np.pi))
            
            # with open(f"{path}/{runname}/macro_pop.csv", "a") as macro_pop:
            #     macro_pop.write(f"{T},{pos_P.shape[0]},{pos_NP.shape[0]}\n")
            
            # np.save(f"{path}/{runname}/pos_P/t{int(n/save_step)}-{T}.npy", pos_P)
            # np.save(f"{path}/{runname}/pos_NP/t{int(n/save_step)}-{T}.npy", pos_NP)
            # np.save(f"{path}/{runname}/v_pos_P/t{int(n/save_step)}-{T}.npy", v_pos_P)
            # np.save(f"{path}/{runname}/v_pos_NP/t{int(n/save_step)}-{T}.npy", v_pos_NP)
        
            t.append(T)
            npart_P.append(pos_P.shape[0])
            npart_NP.append(pos_NP.shape[0])
        
            plt.subplots(1, 2, figsize=(10,5))
            # plt.figure(figsize=(10,10))
            plt.subplot(1, 2, 1)
            plt.scatter(*pos_P.T, c="b", s=1)
            plt.scatter(*pos_NP.T, c="r", s=1)
            plt.scatter(*pvs[strengths>0].T, facecolors="w", edgecolors="k", s=20)
            plt.scatter(*pvs[strengths<0].T, c="k", s=20)
            plt.xlim(0,L)
            plt.ylim(0,L)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 2, 2)
            plt.plot(t, npart_P, c="b", lw=1)
            plt.plot(t, npart_NP, c="r", lw=1)
            # plt.suptitle(f"T = {T:.3f}: {pos.shape[0]} particles")
            plt.show()

        if pos_P.shape[0] == 0 or pos_NP.shape[0] == 0:

            v_pos_P, v_pos_NP = pv_field_particles(pvs, pos_P, pos_NP, strengths, bounds, periodic_repeats)
            
            # with open(f"{path}/{runname}/macro_pop.csv", "a") as macro_pop:
            #     macro_pop.write(f"{T},{pos_P.shape[0]},{pos_NP.shape[0]}\n")
            
            # np.save(f"{path}/{runname}/pos_P/t{int(n/save_step)}-{T}.npy", pos_P)
            # np.save(f"{path}/{runname}/pos_NP/t{int(n/save_step)}-{T}.npy", pos_NP)
            # np.save(f"{path}/{runname}/v_pos_P/t{int(n/save_step)}-{T}.npy", v_pos_P)
            # np.save(f"{path}/{runname}/v_pos_NP/t{int(n/save_step)}-{T}.npy", v_pos_NP)
            
            break

# f = open(f"{path}/{runname}/finished", "w")
# f.close()
