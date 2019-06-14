#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for gradient systems
"""

# numerics imports
import math
import numpy as np
from progressbar import progressbar

class GradientSystem(object):
    
    def __init__(self, potential, domain, beta):
        self.pot = potential.pot
        self.gradPot = potential.gradPot
        self.domain = domain
        self.dimension = np.size(domain,1)
        self.beta = beta
    
    def computeTrajectory(self, t, dt, r0): # compute trajectory starting at first entry in r0; all other entries are ignored
        nsteps = math.floor(t/dt)
        rnew = np.array([r0[0,:]])
        sysdim = np.size(rnew)
        rout = np.empty((nsteps+1,sysdim))
        rout[0,:] = rnew
        
        # parameters for Brownian motion
        mean = np.zeros(sysdim)
        var = (dt*2/self.beta)*np.eye(sysdim)        
        
        for i in progressbar(range(nsteps)):
            nablaV = self.gradPot(rnew) 
            
            #draw Brownian motion
            dW = np.random.multivariate_normal(mean, var)
            
            #update position
            rnew = rnew - dt*nablaV + dW
            rout[i,:] = rnew
        return rout
    
    def computeBurst(self, t, dt, r0, showprogress=True): # parallel computation of trajectory endpoints starting in all entries of r0
        nsteps = math.floor(t/dt)
        npoints = np.size(r0,0)
        sysdim = np.size(r0,1)
        
        # pre-draw the Brownian motion
        mean = np.zeros(sysdim)
        var = (dt*2/self.beta)*np.eye(sysdim)        
        #dW = np.random.multivariate_normal(mean, var, nsteps*npoints).reshape((npoints,sysdim,-1))
        
        print("Sampling transition densities...")
        for i in progressbar(range(nsteps)):
            nablaV = self.gradPot(r0)
            
            #draw Brownian motion
            dW = np.random.multivariate_normal(mean, var, npoints)
            
            #update position
            r0 = r0 - dt*nablaV + dW
            #r0 = r0 - dt*nablaV + dW[:,:,i]
        return r0
    
    def generateTestpoints(self, n, dist='uniform'):
        if dist == 'uniform':
            x = np.random.uniform(self.domain[0], self.domain[1], (n,self.dimension))
            return x
        elif dist == 'grid':
            Xls = []
            for i in range(self.dimension):
                Xls.append(np.linspace(self.domain[0][i], self.domain[1][i], n))
            X = np.meshgrid(*Xls)
            X = np.asarray(X)
            xtest = np.vstack([X[0].ravel(), X[1].ravel()]).transpose()
            return xtest
        