#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manifold learning methods
"""

# numerics imports
import numpy as np
import scipy
import scipy.sparse.linalg as sla
from sklearn.neighbors.kde import KernelDensity
from scipy.integrate import dblquad


def diffusionMaps(xtest, distMat=None, neigs=10, epsi=1., alpha=0.5):
    npoints = np.size(xtest,0)
    
    # compute the distance matrix if it does not exist
    if distMat is None:
        distMat = scipy.spatial.distance.cdist(xtest,xtest)
        
    kernelMat = np.exp(-distMat**2/epsi)
    rowsum  = np.sum(kernelMat,axis=0)
    
    # compensating for testpoint density
    kernelMat = kernelMat / np.outer(rowsum**alpha,rowsum**alpha)
    
    # row normalization
    kernelMat = kernelMat / np.tile(sum(kernelMat,0),(npoints,1))
    
    # weight matrix
    weightMat = np.diag(sum(kernelMat,0))
    
    # solve the diffusion maps eigenproblem
    eigs = sla.eigs(kernelMat, neigs, weightMat)
    return eigs



def L2distance(system, cloud1, cloud2, rho, epsi):
    # 1/rho-weighted L2 distance between densities represented by point clouds
    
    # Compute Kernel density estimate from point clouds
    KDE1 = KernelDensity(kernel="gaussian", bandwidth=epsi).fit(cloud1)
    KDE2 = KernelDensity(kernel="gaussian", bandwidth=epsi).fit(cloud2)
    
    kde1fun = lambda x, y: np.exp(KDE1.score_samples(np.array([[x,y]])))
    kde2fun = lambda x, y: np.exp(KDE2.score_samples(np.array([[x,y]])))
    
    integrand = lambda x, y: (kde1fun(x,y) - kde2fun(x,y))**2 / rho(x,y)

    dist = dblquad(integrand, system.domain[0,0], system.domain[1,0], system.domain[0,1], system.domain[1,1])
    
    return dist