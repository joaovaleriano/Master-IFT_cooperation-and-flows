#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 09:07:21 2022

@author: valeriano
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

scales = 10**np.arange(-1, 4.5, 0.5)

b = 0.15
d = 0.09
a = 0.013
R = 0.05
N = 587

n_pv = 10

seeds = np.arange(5)

plt.figure(figsize=(10,7))
# axins = plt.gca().inset_axes((0.55,1-0.4,.45,.4))
c = 0
for scale in scales[::2]:
    path = f"/home/valeriano/Documents/Masters/abm_rd_results/b{b}d{d}a{a}R{R}scale{scale:.2f}N{N}pv{n_pv}/"
    files = [file for file in os.listdir(path) if ".h5" in file]
    
    pcf_mean = []
    pcf_std = []
    
    for file in files:
        f = h5py.File(path+file, "r")
        pcf_mean.append(np.array(f.get("pcf_mean")))
        pcf_std.append(np.array(f.get("pcf_std")))
    
    pcf_mean_m = np.mean(pcf_mean, axis=0)
    pcf_std_m = np.mean(pcf_std, axis=0)
    
    r_bins = np.linspace(0, R, 41)
    r_bin_mids = (r_bins[1:]+r_bins[:-1])/2
    r_max = r_bins[-1]
    
    ntheta = 41
    theta = np.linspace(0, 2*np.pi, ntheta)
    dtheta = theta[1]-theta[0]
    theta_bin_mids = (theta[1:]+theta[:-1])/2
    
    # for i in range(len(pcf_mean)):
    #     plt.plot(r_bin_mids[1:], np.sum(pcf_mean[i], axis=1)[1:]/99)
    # plt.hlines(1, 0, R, "gray", "--")
    # plt.show()
    
    plt.plot(r_bin_mids, np.sum(pcf_mean_m, axis=1)/40, lw=3, label=f"Da={f.attrs['Da']:.2e}", c=f"C{c}")
    c += 2
    # plt.fill_between(r_bin_mids[1:], np.sum(pcf_mean_m-pcf_std_m, axis=1)[1:]/40, np.sum(pcf_mean_m+pcf_std_m, axis=1)[1:]/40, alpha=0.5)
    # plt.hlines(1, 0, R, "gray", "--")
    # axins.plot(r_bin_mids[:6], np.sum(pcf_mean_m, axis=1)[:6]/40, lw=3, label=f"Da={f.attrs['Da']:.2e}")
plt.xlim(r_bin_mids[0], r_bin_mids[-1])
plt.legend(loc=(0.15, 0.35), fontsize=16)
plt.legend(loc="center right", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"$r$", fontsize=16)
plt.ylabel(r"PCF$(r)$", fontsize=16)
# axins.set_xlim(r_bin_mids[0], r_bin_mids[5])
# plt.show()
plt.savefig("PCF_r.png", dpi=300, bbox_inches="tight")
plt.close()

# plt.subplots(subplot_kw=dict(projection='polar'), figsize=(12,10))
# for scale in scales[::2]:
#     path = f"/home/valeriano/Documents/Masters/abm_rd_results/b{b}d{d}a{a}R{R}scale{scale:.2f}N{N}pv{n_pv}/"
#     files = [file for file in os.listdir(path) if ".h5" in file]
    
#     pcf_mean = []
#     pcf_std = []
    
#     for file in files:
#         f = h5py.File(path+file, "r")
#         pcf_mean.append(np.array(f.get("pcf_mean")))
#         pcf_std.append(np.array(f.get("pcf_std")))
    
#     pcf_mean_m = np.mean(pcf_mean, axis=0)
#     pcf_std_m = np.mean(pcf_std, axis=0)
    
#     r_bins = np.linspace(0, R, 41)
#     r_bin_mids = (r_bins[1:]+r_bins[:-1])/2
#     r_max = r_bins[-1]
    
#     ntheta = 41
#     theta = np.linspace(0, 2*np.pi, ntheta)
#     dtheta = theta[1]-theta[0]
#     theta_bin_mids = (theta[1:]+theta[:-1])/2
    
#     theta2 = np.concatenate((theta_bin_mids, theta_bin_mids[-1:]+dtheta))
#     pcf_mean_m2 = np.concatenate((pcf_mean_m, pcf_mean_m[:,0:1]), axis=1)
#     pcf_std_m2 = np.concatenate((pcf_std_m, pcf_std_m[:,0:1]), axis=1)
    
#     plt.plot(theta2, np.sum(pcf_mean_m2, axis=0)/(ntheta-1), lw=3, label=f"Da={f.attrs['Da']:.2e}")
#     # plt.fill_between(theta2, np.sum(pcf_mean_m2-pcf_std_m2, axis=0)/(ntheta-1), 
#     #                   np.sum(pcf_mean_m2+pcf_std_m2, axis=0)/(ntheta-1), alpha=0.5)

# plt.legend(loc=(1.01,0.56), fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks([])
# plt.xlabel(r"$\theta$", fontsize=16)
# plt.title(r"PCF$(\theta)$", fontsize=20)
# # plt.show()
# plt.savefig("PCF_theta_polar.png", dpi=300, bbox_inches="tight")
# plt.close()

# plt.figure(figsize=(10,7))
# for scale in scales:
#     path = f"/home/valeriano/Documents/Masters/abm_rd_results/b{b}d{d}a{a}R{R}scale{scale:.2f}N{N}pv{n_pv}/"
#     files = [file for file in os.listdir(path) if ".h5" in file]
    
#     pcf_mean = []
#     pcf_std = []
    
#     for file in files:
#         f = h5py.File(path+file, "r")
#         pcf_mean.append(np.array(f.get("pcf_mean"))[1:])
#         pcf_std.append(np.array(f.get("pcf_std"))[1:])
    
#     pcf_mean_m = np.mean(pcf_mean, axis=0)
#     pcf_std_m = np.mean(pcf_std, axis=0)
    
#     r_bins = np.linspace(0, R, 41)
#     r_bin_mids = (r_bins[1:]+r_bins[:-1])[1:]/2
#     r_max = r_bins[-1]
    
#     ntheta = 41
#     theta = np.linspace(0, 2*np.pi, ntheta)
#     dtheta = theta[1]-theta[0]
#     theta_bin_mids = (theta[1:]+theta[:-1])/2
    
#     theta2 = np.concatenate((theta_bin_mids, theta_bin_mids[-1:]+dtheta))
#     pcf_mean_m2 = np.concatenate((pcf_mean_m, pcf_mean_m[:,0:1]), axis=1)
#     pcf_std_m2 = np.concatenate((pcf_std_m, pcf_std_m[:,0:1]), axis=1)
    
#     plt.plot(theta2, np.sum(pcf_mean_m2, axis=0)/(ntheta-1), lw=3, label=f"Da={f.attrs['Da']:.0e}")
#     # plt.fill_between(theta2, np.sum(pcf_mean_m2-pcf_std_m2, axis=0)/(ntheta-1), 
#     #                  np.sum(pcf_mean_m2+pcf_std_m2, axis=0)/(ntheta-1), alpha=0.5)
# plt.xlim(theta2[0], theta2[-1])
# plt.legend(loc=(1.01, 0.3), fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel(r"$\theta$", fontsize=16)
# plt.ylabel(r"PCF$(\theta)$", fontsize=16)
# # plt.show()
# plt.savefig("PCF_theta.png", dpi=300, bbox_inches="tight")
# plt.close()

# plt.figure(figsize=(10,7))
# for scale in scales:
#     path = f"/home/valeriano/Documents/Masters/abm_rd_results/b{b}d{d}a{a}R{R}scale{scale:.2f}N{N}pv{n_pv}/"
#     files = [file for file in os.listdir(path) if ".h5" in file]
    
#     pcf_mean = []
#     pcf_std = []
    
#     for file in files:
#         f = h5py.File(path+file, "r")
#         pcf_mean.append(np.array(f.get("pcf_mean"))[1:])
#         pcf_std.append(np.array(f.get("pcf_std"))[1:])
    
#     pcf_mean_m = np.mean(pcf_mean, axis=0)
#     pcf_std_m = np.mean(pcf_std, axis=0)
    
#     r_bins = np.linspace(0, R, 41)
#     r_bin_mids = (r_bins[1:]+r_bins[:-1])[1:]/2
#     r_max = r_bins[-1]
    
#     ntheta = 41
#     theta = np.linspace(0, 2*np.pi, ntheta)
#     dtheta = theta[1]-theta[0]
#     theta_bin_mids = (theta[1:]+theta[:-1])/2
    
#     theta2 = np.concatenate((theta_bin_mids, theta_bin_mids[-1:]+dtheta))
#     pcf_mean_m2 = np.concatenate((pcf_mean_m, pcf_mean_m[:,0:1]), axis=1)
#     pcf_std_m2 = np.concatenate((pcf_std_m, pcf_std_m[:,0:1]), axis=1)
    
#     Da = f.attrs['Da']
    
#     ecc = np.sum((np.sum(pcf_mean_m2, axis=0)/(ntheta-1))[np.abs(np.tan(theta2))<=1])/np.sum((np.sum(pcf_mean_m2, axis=0)/(ntheta-1))[np.abs(np.tan(theta2))>=1])-1
    
#     plt.scatter(Da, ecc, c="k", s=50)

# plt.xscale("log")
# # plt.legend(loc="center right", fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel(r"$Da$", fontsize=16)
# plt.ylabel(r"Velocity alignment of PCF($\theta$)", fontsize=16)
# # plt.show()
# plt.savefig("v_alignment.png", dpi=300, bbox_inches="tight")
# plt.close()

# # plt.figure(figsize=(10,7))
# # for scale in scales:
# #     path = f"/home/valeriano/Documents/Masters/abm_rd_results/b{b}d{d}a{a}R{R}scale{scale:.2f}N{N}pv{n_pv}/"
# #     files = [file for file in os.listdir(path) if ".h5" in file]
    
# #     npart = []
    
# #     for file in files:
# #         f = h5py.File(path+file, "r")
# #         npart.append(np.array(f.get("npart"))[1:])
    
# #     # npart = np.array(npart)
# #     npart_mean = np.mean(npart, axis=0)
# #     npart_std = np.std(npart, axis=0)
# #     # pcf_std_m = np.mean(pcf_std, axis=0)
    
# #     Da = f.attrs['Da']
    
# #     plt.plot(npart_mean, lw=3, label=f"Da={f.attrs['Da']:.2e}")

# # plt.xlim(0, len(npart_mean))
# # # plt.legend(loc="center right", fontsize=16)
# # plt.xticks(fontsize=14)
# # plt.yticks(fontsize=14)
# # plt.xlabel(r"$t$", fontsize=16)
# # plt.ylabel(r"N", fontsize=16)
# # plt.show()

# plt.figure(figsize=(10,7))
# for scale in scales:
#     path = f"/home/valeriano/Documents/Masters/abm_rd_results/b{b}d{d}a{a}R{R}scale{scale:.2f}N{N}pv{n_pv}/"
#     files = [file for file in os.listdir(path) if ".h5" in file]
    
#     npart = []
    
#     for file in files:
#         f = h5py.File(path+file, "r")
#         npart.append(np.array(f.get("npart"))[1:])
    
#     # npart = np.array(npart)
#     npart_mean = np.mean(npart, axis=0)
#     npart_std = np.std(npart, axis=0)
#     # pcf_std_m = np.mean(pcf_std, axis=0)
    
#     Da = f.attrs['Da']
    
#     # plt.plot(npart_mean, lw=3, label=f"Da={f.attrs['Da']:.2e}")
#     plt.errorbar(Da, np.mean(npart_mean), np.mean(npart_std), lw=3, capsize=10, fmt="o", ms=10, c="k")
    
# path = f"/home/valeriano/Documents/Masters/abm_rd_results/b{b}d{d}a{a}R{R}N{N}/"
# files = [file for file in os.listdir(path) if ".h5" in file]

# npart = []

# for file in files:
#     f = h5py.File(path+file, "r")
#     npart.append(np.array(f.get("npart"))[1:])

# # npart = np.array(npart)
# npart_mean = np.mean(npart, axis=0)
# npart_std = np.std(npart, axis=0)
# # pcf_std_m = np.mean(pcf_std, axis=0)

# Da = f.attrs['Da']

# # plt.plot(npart_mean, lw=3, label=f"Da={f.attrs['Da']:.2e}")
# # plt.hlines(np.mean(npart_mean), 1e-3, 1e1, color="gray", ls="--", lw=3)

# plt.xscale("log")
# # plt.legend(loc="center right", fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel(r"$Da$", fontsize=16)
# plt.ylabel(r"$\langle N \rangle$", fontsize=16)
# # plt.show()
# plt.savefig("N_vs_Da.png", dpi=300, bbox_inches="tight")
# plt.close()