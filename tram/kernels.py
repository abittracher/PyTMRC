#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various kernels used for RKHS-embedding and Diffusion Maps algorithm
"""

# numerics imports
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

# utility imports
from tqdm import tqdm

from scipy.spatial.distance import cdist

class Kernel:
    
    def computeMercerEigs(self, xtest, neigs):
        GxxTest = self.evaluate(xtest,xtest)
        eigs = sla.eigs(GxxTest,neigs)
        return eigs
    
    def computeMercerProjection(self, r, xtest, neigs, eigs=None, showprogress=True):
        GxxTest = self.evaluate(xtest,xtest) # Gram matrix in the eigenvector test points
        GxxBurst = self.evaluate(r,xtest) # Gram matrix in the trajectory end points
        pGxxTest = la.pinv(GxxTest)
        
        # if no eigenfunctions are provided, compute them
        if not(eigs):
            eigs = self.computeMercerEigs(xtest, neigs)
            
        h = []
        # inner product between the density implied by the point cloud and the i-th eigenfunction
        for i in tqdm(range(neigs), disable = not showprogress):
            y = (GxxBurst.dot(pGxxTest)).dot(eigs[1][:,i])
            h = np.append(h, np.sum(y)) # the sum is MonteCarlo quadrature
        #return h
        # TODO: The following computation of the compression factor is not working correctly
        c = []
        # compression factor lower bound
        for i in range(neigs):
            ci = 1 + np.sum(h[(i+1):]**2)/np.sum(h[:(i+1)]**2)
            np.append(c, np.sqrt(eigs[0][i]/ci))
        return [h,c]
        
    def pointcloudDist(self, X, Y):
        npointsX = np.size(X,0)
        npointsY = np.size(Y,0)
        GXX = self.evaluate(X,X)
        GYY = self.evaluate(Y,Y)
        GXY = self.evaluate(X,Y)
        
        d = np.sum(GXX)/npointsX**2 + np.sum(GYY)/npointsY**2 - 2*np.sum(GXY)/(npointsX*npointsY)
        return d
        

class GaussianKernel(Kernel):
    def __init__(self, epsi=1.):
        self.epsi = epsi # Kernel bandwidth
        
    def evaluate(self, x, y):
    # evaluates the kernel at all combinations of entries of x and y         
        distmat = cdist(x, y, 'sqeuclidean')
        kmat =  np.exp(-distmat/self.epsi)
        return kmat
    
    
class PolynomialKernel(Kernel):
    def __init__(self, p=2):
        self.p = p # Polynomial degree
        
    def evaluate(self, x, y):
    # evaluates the kernel at all combinations of entries of x and y         
        kmat =  (1 + np.matmul(x,y.transpose()))**self.p
        return kmat
