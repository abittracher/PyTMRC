#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transition Manifold-related classes and methods
"""

# numerics imports
import numpy as np
from progressbar import progressbar
from scipy.spatial import cKDTree
from scipy.ndimage.interpolation import shift
from sklearn.neighbors.kde import KernelDensity
from scipy.integrate import dblquad

# TM imports
import tram.manifold_learning as ml


class TransitionManifold:

    rc = None

    def __init__(self, system, xtest):
        self.system = system
        self.xtest = xtest


# TM based on RKHS-embeddings of parallel short simulations
class KernelBurstTransitionManifold(TransitionManifold):

    def __init__(self, system, kernel, xtest, t, dt, M,  epsi=1.):
        self.system = system
        self.kernel = kernel
        self.xtest = xtest
        self.t = t
        self.dt = dt
        self.M = M
        self.epsi = epsi

    def computeRC(self):
        npoints = np.size(self.xtest,0)

        # compute the time evolution of all test points at once, for performance reasons
        x0 = np.tile(self.xtest, (self.M,1))
        pointclouds = self.system.computeBurst(self.t, self.dt, x0, showprogress=True)

        # compute symmetric kernel evaluations
        dXX = []
        print("Computing symmetric kernel evaluations...")
        for i in progressbar(range(npoints)):
            GXX = self.kernel.evaluate(pointclouds[i::npoints,:], pointclouds[i::npoints,:])
            dXX = np.append(dXX, np.sum(GXX))

        # compute asymmetric kernel evaluations and assemble distance matrix
        distMat = np.zeros((npoints, npoints))
        print("Computing asymmetric kernel evaluations...")
        for i in progressbar(range(npoints)):
            for j in range(i):
                GXY = self.kernel.evaluate(pointclouds[i::npoints,:], pointclouds[j::npoints,:])
                distMat[i,j] = (dXX[i] + dXX[j] - 2*np.sum(GXY)) / self.M**2
        distMat = distMat + np.transpose(distMat)

        # compute diffusion maps coordinates
        eigs = ml.diffusionMaps(self.xtest, distMat, epsi=self.epsi)
        self.rc = eigs
        self.distMat = distMat




# TM based on RKHS-embeddings of a single long trajectory
class KernelTrajTransitionManifold(TransitionManifold):

    def __init__(self, system, kernel, xtest, traj, lag, epsi=1.):
        self.system = system
        self.kernel = kernel
        self.xtest = xtest
        self.traj = traj
        self.lag = lag
        self.epsi = epsi

    def computeRC(self):
        npoints = np.size(self.xtest,0)

        # indices of test points closest to trajectory points
        kdTree = cKDTree(self.xtest)
        closest = kdTree.query(self.traj, n_jobs=-1)[1]

        # extract point clouds from trajectory
        pointclouds = []
        print("Assigning trajectory points to centers...")
        for i in progressbar(range(npoints)):
            laggedInd = shift(closest==i, self.lag, cval=False) # indices of lagged points
            pointclouds.append(self.traj[laggedInd,:])

        # compute symmetric kernel evaluations
        dXX = []
        print("Computing symmetric kernel evaluations...")
        for i in progressbar(range(npoints)):
            GXX = self.kernel.evaluate(pointclouds[i], pointclouds[i])
            dXX = np.append(dXX, np.sum(GXX))

        # compute asymmetric kernel evaluations and assemble distance matrix
        distMat = np.zeros((npoints, npoints))
        print("Computing asymmetric kernel evaluations...")
        for i in progressbar(range(npoints)):
            nTrajpointsi = np.size(pointclouds[i],0)
            for j in range(i):
                nTrajpointsj = np.size(pointclouds[j],0)
                GXY = self.kernel.evaluate(pointclouds[i], pointclouds[j])
                distMat[i,j] = dXX[i]/nTrajpointsi**2 + dXX[j]/nTrajpointsj**2 - 2*np.sum(GXY)/(nTrajpointsi*nTrajpointsj)
        distMat = distMat + np.transpose(distMat)

        eigs = ml.diffusionMaps(self.xtest, distMat, epsi=self.epsi)
        self.rc = eigs
        self.distMat = distMat



# TM based on random Whitney embeddings of parallel short simulations
class EmbeddingBurstTransitionManifold(TransitionManifold):

    def __init__(self, system, embfun, xtest, t, dt, M, epsi=1.):
        self.system = system
        self.embfun = embfun
        self.xtest = xtest
        self.t = t
        self.dt = dt
        self.M = M
        self.epsi = epsi

    def computeRC(self):
        npoints = np.size(self.xtest,0)

        # compute the time evolution of all test points at once, for performance reasons
        x0 = np.tile(self.xtest, (self.M,1))
        pointclouds = self.system.computeBurst(self.t, self.dt, x0, showprogress=True)

        # embedd each point cloud into R^k
        embpointclouds = np.zeros((0,(self.embfun).outputdimension))
        for i in range(npoints):
            y = self.embfun.evaluate(pointclouds[i::npoints,:])
            embpointclouds = np.append(embpointclouds, [np.sum(y,0)/self.M], axis=0)
        self.embpointclouds = embpointclouds

        # compute diffusion maps coordinates on embedded points
        eigs = ml.diffusionMaps(embpointclouds, epsi=self.epsi)
        self.rc= eigs



# random linear embedding function for the Whitney embedding
class RandomLinearEmbeddingFunction():

    def __init__(self, inputdimension, outputdimension, seed):
        self.inputdimension = inputdimension
        self.outputdimension = outputdimension
        self.seed = seed

        # draw the random coefficients
        np.random.seed(self.seed)
        A = np.random.uniform(0, 1, (self.inputdimension,self.outputdimension))
        #self.A,_ = np.linalg.qr(A,mode='complete')
        self.A = A


    def evaluate(self, x):
        y = x.dot(self.A)
        return y



# TM based on direct L2-distance comparison between densities represented by parallel shor simulations
class L2BurstTransitionManifold(TransitionManifold):

    def __init__(self, system, rho, xtest, t, dt, M, epsi=1., kde_epsi=0.1):
        self.system = system
        self.rho = rho
        self.xtest = xtest
        self.t = t
        self.dt = dt
        self.M = M
        self.epsi = epsi
        self.kde_epsi = kde_epsi

    def L2distance(self, cloud1, cloud2):
        # 1/rho-weighted L2 distance between densities represented by point clouds

        KDE1 = KernelDensity(kernel="gaussian", bandwidth=self.kde_epsi).fit(cloud1)
        KDE2 = KernelDensity(kernel="gaussian", bandwidth=self.kde_epsi).fit(cloud2)

        kde1fun = lambda x, y: np.exp(KDE1.score_samples(np.array([[x,y]])))
        kde2fun = lambda x, y: np.exp(KDE2.score_samples(np.array([[x,y]])))

        integrand = lambda x, y: (kde1fun(x,y) - kde2fun(x,y))**2 / self.rho(x,y)

        dist = dblquad(integrand, self.system.domain[0,0], self.system.domain[1,0], self.system.domain[0,1], self.system.domain[1,1])
        return dist

    def computeRC(self):
        npoints = np.size(self.xtest,0)

        # compute the time evolution of all test points at once, for performance reasons
        x0 = np.tile(self.xtest, (self.M,1))
        pointclouds = self.system.computeBurst(self.t, self.dt, x0, showprogress=True)

        # compute distance matrix
        distMat = np.zeros((npoints, npoints))
        print("Computing distance matrix...")
        for i in progressbar(range(npoints)):
            for j in range(npoints):
                distMat[i,j] = self.L2distance(pointclouds[i::npoints,:], pointclouds[j::npoints,:])[0]
        self.distMat = distMat

        # compute diffusion maps coordinates on embedded points
        #eigs = ml.diffusionMaps(embpointclouds, epsi=self.epsi)
        #self.rc= eigs

