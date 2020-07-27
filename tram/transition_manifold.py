#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transition Manifold-related classes and methods.
"""

#TODO: create base class for diffusion map based TMs,
#unify fit/predict method

# system imports
import sys

# numerics imports
import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
from scipy.ndimage.interpolation import shift
from sklearn.neighbors.kde import KernelDensity
from scipy.integrate import dblquad
from sklearn.kernel_approximation import RBFSampler, Nystroem

# utility imports
from tqdm import tqdm

# TM imports
import tram.manifold_learning as ml

class TransitionManifold:
    """
    TransitionManifold base class
    """
    def __init__(self):
        self._fitted = False

    def fit(self, X):
        self._fitted = True

    def predict(self, Y):
        if not self._fitted:
            raise RuntimeError("Run fit() method first before predicting")


# TM based on RKHS-embeddings of parallel short simulations
class KernelBurstTransitionManifold(TransitionManifold):
    """
    Kernel transition manifold based on the kernel mean embeddings
    of transition densities and maximum mean discrepancy.
    The reaction coordinate is computed by diffusion maps
    based on the embedded empirical approximations of the transition
    densities.
    This estimator is based on multiple iid simulations (bursts) of the dynamics
    such that pointclouds can be used as empirical approximations
    of transition densities corresponding to Dirac functions
    in state space.

    Please refer to 
    A. Bittracher, S. Klus, B. Hamzi, C. Schütte
    'A kernel-based method for coarse graining complex dynamical systems' 
    https://arxiv.org/abs/1904.08622
    """

    #TODO rename epsi --> gamma
    def __init__(self, kernel, epsi=1.):
        """
        Instantiates a kernel transition manifold estimator.

        Parameters
        ----------
        kernel : tram.kernels.Kernel object
        epsi : float, bandwidth of the distance kernel used to assemble the
            similarity matrix for diffusion maps.   
            NOTE: This is NOT the bandwidth of the reproducing kernel used
            for the RKHS embedding of the transition densities.

        Example:
        >>> kernel = tram.kernels.GaussianKernel()
        >>> kernel_tm = KernelBurstTransitionManifold(kernel, epsi=1.)
        """
        super().__init__()
        self.kernel = kernel
        self.epsi = epsi

    def fit(self, X, n_components=10, showprogress = True):
        """
        Computes the reaction coordinate based on the data X.

        Parameters
        ----------
        X : np.array of shape [# startpoints, # simulations per startpoint, dimension]
            data array containing endpoints of trajectory simulations for each startpoint
        n_components : int, number of the eigenpairs of the diffusion matrix
            which are computed.
            NOTE: n_components needs to be at least as big as the dimension
            of the desired reaction coordinate to which the data is projected by the
            .predict() method.
        """
        super().fit(X)
        
        #TODO update computational routine to new interface
        npoints, self.M = X.shape[:2]
        X = _reshape(X)

        # compute symmetric kernel evaluations
        dXX = []
        print("Computing symmetric kernel evaluations...")
        sys.stdout.flush() # workaround for messed-up progress bars
        for i in tqdm(range(npoints), disable = not showprogress):
            GXX = self.kernel.evaluate(X[i::npoints,:], X[i::npoints,:])
            dXX = np.append(dXX, np.sum(GXX))

        # compute asymmetric kernel evaluations and assemble distance matrix
        distMat = np.zeros((npoints, npoints))
        print("Computing asymmetric kernel evaluations...")
        sys.stdout.flush() # workaround for messed-up progress bars
        for i in tqdm(range(npoints), disable = not showprogress):
            for j in range(i):
                GXY = self.kernel.evaluate(X[i::npoints,:], X[j::npoints,:])
                distMat[i,j] = (dXX[i] + dXX[j] - 2*np.sum(GXY)) / self.M**2
        distMat = distMat + np.transpose(distMat)

        # compute diffusion maps coordinates
        eigs = ml.diffusionMaps(distMat, epsi=self.epsi, n_components=n_components)
        self.rc = eigs
        self.distMat = distMat

    def predict(self, n_components):
        """
        Project data to diffusion space
        
        NOTE: the maximal possible dimension of the diffusion space is 
            determined by the number of eigenpairs available specified 
            by the argument <n_components> in the .fit() routine
        NOTE: this method returns all dimensions in diffusion space including the 
            dimension related to first eigenpair. When projecting to diffusion space,
            it is therefore reasonable to not use the first coordinate 
            returned by this method.
        
        Example
        ----------
        Projecting to a one-dimensional reaction coordinate
            
            >>> transformed = kerTM.predict(n_components=2)
            >>> reaction_coordinate = transformed[:, 1]

        Parameters
        ----------
        n_components: int, number of dimensions in
            diffusion space

        Returns
        -------
        array of shape (n_features, n_components)
            the transformed data in diffusion space  
        """
        super().predict(None)
        return ml.evaluateDiffusionMaps(self.rc, n_components+1)[:,1::]


# TM based on RKHS-embeddings of a single long trajectory
class KernelTrajTransitionManifold(TransitionManifold):
    """
    Kernel transition manifold based on the kernel mean embeddings
    of evolved indicator densities over Voronoi cells. If the diameters of
    the cells are small, these densities approximate the transition 
    densities starting from the cell center points.
    This estimator is based on a single long trajectory and approximates
    the evolved indicator densities by considering all trajectory points
    within one Voronoi cell and treating the shifted points within the 
    trajectory as endpoints of short simulations.

    Consistency analysis of this method is current work in progress.
    The method should thus be considered experimental.
    """

    #TODO rename epsi --> gamma
    def __init__(self, kernel, epsi=1.):
        """
        Instantiates a kernel transition manifold estimator.

        Parameters
        ----------
        kernel : tram.kernels.Kernel object
        epsi : float, bandwidth of the distance kernel used to assemble the
            similarity matrix for diffusion maps.   
            NOTE: This is NOT the bandwidth of the reproducing kernel used
            for the RKHS embedding of the transition densities.

        Example:
        >>> kernel = tram.kernels.GaussianKernel()
        >>> kernel_tm = KernelTrajTransitionManifold(kernel, epsi=1.)
        """
        super().__init__()
        self.kernel = kernel
        self.epsi = epsi

    def fit(self, X, Xtest, lag, n_components=10, showprogress = True):
        """
        Computes the reaction coordinate based on the trajectory data X.

        Parameters
        ----------
        X : np.array of shape [# steps, dimension]
            data array containing steps of a trajectory simulation
        Xtest : np.array of shape [# points, dimension]
            array containing center points of Voronoi cells
        lag : int
            number of steps after which the endpoints are selected
        n_components : int, number of the eigenpairs of the diffusion matrix
            which are computed.
            NOTE: n_components needs to be at least as big as the dimension
            of the desired reaction coordinate to which the data is projected by the
            .predict() method.
        """
        super().fit(X)
        
        npoints = np.size(Xtest,0)

        # indices of test points closest to trajectory points
        print("Sorting into Voronoi cells...")
        kdTree = cKDTree(Xtest)
        closest = kdTree.query(X, n_jobs=-1)[1]

        # extract point clouds from trajectory
        pointclouds = []
        print("Assigning trajectory points to centers...")
        sys.stdout.flush() # workaround for messed-up progress bars
        for i in tqdm(range(npoints)):
            laggedInd = shift(closest==i, lag, cval=False) # indices of lagged points
            pointclouds.append(X[laggedInd,:])

        # compute symmetric kernel evaluations
        dXX = []
        print("Computing symmetric kernel evaluations...")
        sys.stdout.flush() # workaround for messed-up progress bars
        for i in tqdm(range(npoints), disable = not showprogress):
            GXX = self.kernel.evaluate(pointclouds[i], pointclouds[i])
            dXX = np.append(dXX, np.sum(GXX))

        # compute asymmetric kernel evaluations and assemble distance matrix
        distMat = np.zeros((npoints, npoints))
        print("Computing asymmetric kernel evaluations...")
        sys.stdout.flush() # workaround for messed-up progress bars
        for i in tqdm(range(npoints), disable = not showprogress):
            nTrajpointsi = np.size(pointclouds[i],0)
            for j in range(i):
                nTrajpointsj = np.size(pointclouds[j],0)
                GXY = self.kernel.evaluate(pointclouds[i], pointclouds[j])
                distMat[i,j] = dXX[i]/nTrajpointsi**2 + dXX[j]/nTrajpointsj**2 - 2*np.sum(GXY)/(nTrajpointsi*nTrajpointsj)
        distMat = distMat + np.transpose(distMat)

        eigs = ml.diffusionMaps(distMat, epsi=self.epsi)
        self.rc = eigs
        self.distMat = distMat 

    # TODO: implement evaluation at arbitrary points
    def predict(self, n_components):
        """
        Project data to diffusion space
        
        NOTE: the maximal possible dimension of the diffusion space is 
            determined by the number of eigenpairs available specified 
            by the argument <n_components> in the .fit() routine
        NOTE: this method returns all dimensions in diffusion space including the 
            dimension related to first eigenpair. When projecting to diffusion space,
            it is therefore reasonable to not use the first coordinate 
            returned by this method.
        
        Example
        ----------
        Projecting to a one-dimensional reaction coordinate
            
            >>> transformed = kernel_tm.predict(n_components=2)
            >>> reaction_coordinate = transformed[:, 1]

        Parameters
        ----------
        n_components: int, number of dimensions in
            diffusion space

        Returns
        -------
        array of shape (n_features, n_components)
            the transformed data in diffusion space  
        """
        
        super().predict(None)
        return ml.evaluateDiffusionMaps(self.rc, n_components+1)[:,1::]



# TM based on random Whitney embeddings of parallel short simulations
class EmbeddingBurstTransitionManifold(TransitionManifold):
    """
    Embedding transition manifold class based on the Whitney embedding
    of transition densities and their euclidean distance in embedding space.
    This estimator is based on multiple iid simulations (bursts) of the dynamics
    such that pointclouds can be used as empirical approximations
    of transition densities corresponding to Dirac functions
    in state space.

    Please refer to 
    A. Bittracher, P. Koltai, S. Klus, R. Banisch, M. Dellnitz, C. Schütte
    'Transition Manifolds of Complex Metastable Systems - 
    Theory and Data-Driven Computation of Effective Dynamics'
    J Nonlinear Sci (2018) 28:471–512
    https://doi.org/10.1007/s00332-017-9415-0
    """

    #TODO rename epsi --> gamma
    def __init__(self, embfun, epsi=1.):
        """
        Instantiates a Whitney embedding transition manifold estimator.

        Parameters
        ----------
        embfun : EmbeddingFunction, the embedding function for the Whitney
            embedding
        epsi : float, bandwidth of the distance kernel used to assemble the
            similarity matrix for diffusion maps.   

        Example:
        >>> embfun = tram.embedding_functions.RandomLinearEmbeddingFunction(2, 3, 0)
        >>> emb_tm = EmbeddingBurstTransitionManifold(embfun, epsi=1.)
        """
        super().__init__()
        self.embfun = embfun
        self.epsi = epsi
        
    def embedd(self, X, showprogress=True):
        """
        Embedds the point cloud data in X into Euclidean space via the objects
        own EmbeddingFunction. Stores the resulting points in an array
        self.embpointclouds.
        
        Parameters
        ----------
        X : np.array of shape [# startpoints, # simulations per startpoint, dimension]
            data array containing endpoints of trajectory simulations for each startpoint
        """
        #TODO update computational routine to new interface
        npoints, self.M = X.shape[:2]
        X = _reshape(X)

        # embedd each point cloud into R^k
        embpointclouds = np.zeros((0,(self.embfun).outputdimension))
        print("Evaluating observables...")
        sys.stdout.flush() # workaround for messed-up progress bars
        for i in tqdm(range(npoints), disable = not showprogress):
            y = self.embfun.evaluate(X[i::npoints,:])
            embpointclouds = np.append(embpointclouds, [np.sum(y,0)/self.M], axis=0)
        self.embpointclouds = embpointclouds
        

    def fit(self, X, n_components=10, showprogress=True):
        """
        Computes the reaction coordinate based on the data X.

        Parameters
        ----------
        X : np.array of shape [# startpoints, # simulations per startpoint, dimension]
            data array containing endpoints of trajectory simulations for each startpoint
        n_components : int, number of the eigenpairs of the diffusion matrix
            which are computed.
            NOTE: n_components needs to be at least as big as the dimension
            of the desired reaction coordinate to which the data is projected by the
            .predict() method.
        """
        super().fit(X)

        #TODO update computational routine to new interface
        npoints, self.M = X.shape[:2]
        X = _reshape(X)

        # embedd each point cloud into R^k
        embpointclouds = np.zeros((0,(self.embfun).outputdimension))
        print("Evaluating observables...")
        sys.stdout.flush() # workaround for messed-up progress bars
        for i in tqdm(range(npoints), disable = not showprogress):
            y = self.embfun.evaluate(X[i::npoints,:])
            embpointclouds = np.append(embpointclouds, [np.sum(y,0)/self.M], axis=0)
        self.embpointclouds = embpointclouds

        # compute diffusion maps coordinates on embedded points
        distMat = scipy.spatial.distance.cdist(embpointclouds, embpointclouds)
        eigs = ml.diffusionMaps(distMat, epsi=self.epsi, n_components=10)
        self.rc= eigs
    
    def predict(self, n_components):
        """
        Project data to diffusion space
        
        NOTE: the maximal possible dimension of the diffusion space is 
            determined by the number of eigenpairs available specified 
            by the argument <n_components> in the .fit() routine
        NOTE: this method returns all dimensions in diffusion space including the 
            dimension related to first eigenpair. When projecting to diffusion space,
            it is therefore reasonable to not use the first coordinate 
            returned by this method.
        
        Example
        ----------
        Projecting to a one-dimensional reaction coordinate
            
            >>> transformed = embTM.predict(n_components=2)
            >>> reaction_coordinate = transformed[:, 1]

        Parameters
        ----------
        n_components: int, number of dimensions in
            diffusion space

        Returns
        -------
        array of shape (n_features, n_components)
            the transformed data in diffusion space  
        """
        super().predict(None)
        return ml.evaluateDiffusionMaps(self.rc, n_components+1)[:,1::]


# TM based on Whitney-embeddings of a single long trajectory
class EmbeddingTrajTransitionManifold(TransitionManifold):
    """
    Embedding transition manifold based on the Whitney embedding of 
    evolved indicator densities over Voronoi cells. If the diameters of
    the cells are small, these densities approximate the transition 
    densities starting from the cell center points.
    This estimator is based on a single long trajectory and approximates
    the Euclidean inner product between the embedded evolved indicator densities
    by counting transitions between the Voronoi cells within the trajectory.

    Please refer to 
    A. Bittracher, R. Banisch, C. Schütte
    'Data-driven computation of molecular reaction coordinates'
    J Chem Phys 149, 154103 (2018)
    https://doi.org/10.1063/1.5035183
    """

    #TODO rename epsi --> gamma
    def __init__(self, embfun, epsi=1.):
        """
        Instantiates a Whitney embedding transition manifold estimator.

        Parameters
        ----------
        embfun : tram.embedding_functions.RandomLinearEmbeddingFunction object
        epsi : float, bandwidth of the distance kernel used to assemble the
            similarity matrix for diffusion maps.   

        Example:
        >>> kernel = tram.kernels.GaussianKernel()
        >>> emb_tm = EmbeddingTrajTransitionManifold(embfun, epsi=1.)
        """
        super().__init__()
        self.embfun = embfun
        self.epsi = epsi

    def fit(self, X, Xtest, lag, n_components=10, showprogress = True):
        """
        Computes the reaction coordinate based on the trajectory data X.

        Parameters
        ----------
        X : np.array of shape [# steps, dimension]
            data array containing steps of a trajectory simulation
        Xtest : np.array of shape [# points, dimension]
            array containing center points of Voronoi cells
        lag : int
            number of steps after which the endpoints are selected
        n_components : int, number of the eigenpairs of the diffusion matrix
            which are computed.
            NOTE: n_components needs to be at least as big as the dimension
            of the desired reaction coordinate to which the data is projected by the
            .predict() method.
        """
        super().fit(X)
        
        npoints = np.size(Xtest,0)
        dim = np.size(Xtest,1)

        # indices of test points closest to trajectory points
        print("Sorting into Voronoi cells...")
        kdTree = cKDTree(Xtest)
        closest = kdTree.query(X, n_jobs=-1)[1]

        # embedding trajectory points
        Y = self.embfun.evaluate(X)
        embdim = np.size(Y,1)

        # extract and embedd point clouds
        embpointclouds = np.zeros((npoints,embdim))
        print("Embedding point clouds...")
        sys.stdout.flush() # workaround for messed-up progress bars
        for i in tqdm(range(npoints)):
            laggedInd = shift(closest==i, lag, cval=False) # indices of lagged points
            embpointclouds[i,:] = np.sum(Y[laggedInd,:], axis=0) / np.count_nonzero(laggedInd)
        self.embpointclouds = embpointclouds

        # compute diffusion maps coordinates on embedded points
        distMat = scipy.spatial.distance.cdist(embpointclouds, embpointclouds)
        eigs = ml.diffusionMaps(distMat, epsi=self.epsi, n_components=10)
        self.rc= eigs


    # TODO: implement evaluation at arbitrary points
    def predict(self, n_components):
        """
        Project data to diffusion space
        
        NOTE: the maximal possible dimension of the diffusion space is 
            determined by the number of eigenpairs available specified 
            by the argument <n_components> in the .fit() routine
        NOTE: this method returns all dimensions in diffusion space including the 
            dimension related to first eigenpair. When projecting to diffusion space,
            it is therefore reasonable to not use the first coordinate 
            returned by this method.
        emb
        Example
        ----------
        Projecting to a one-dimensional reaction coordinate
            
            >>> transformed = kernel_tm.predict(n_components=2)
            >>> reaction_coordinate = transformed[:, 1]

        Parameters
        ----------
        n_components: int, number of dimensions in
            diffusion space

        Returns
        -------
        array of shape (n_features, n_components)
            the transformed data in diffusion space  
        """
        
        super().predict(None)
        return ml.evaluateDiffusionMaps(self.rc, n_components+1)[:,1::]


class LinearRandomFeatureManifold(TransitionManifold):
    """
    A class providing a linear transition manifold
    by using kernel feature Approximations. The kernel embeddings
    of the transition densities are approximated with
    either random Fourier features or the Nystroem method.
    This estimator is based on multiple iid simulations (bursts) of the dynamics
    such that pointclouds can be used as empirical approximations
    of transition densities corresponding to Dirac functions
    in state space.

    Please refer also to the documentation of sklearn.kernel_approximation
    """

    def __init__(self, method="rff", n_components=100, kernel="rbf", gamma=.1, **kwargs):
        """
        TODO document interface
        TODO add output dimension choice to PCA routine

        Parameters
        ----------
        method : str
            specifies the used feature approximation. Can either be 'rff' or 'nystroem'
        n_components : int
            number of dimensions in the feature approximation space.
        """
        super().__init__()
        self.method = method
        self.gamma = gamma
        self.kernel = kernel
        self.n_components = n_components
        self.kwargs = kwargs

        self.sampler = None
        self.embedded = None
        self.vec = None

    def fit(self, X):
        """
        Computes a linear reaction coordinate based on the data X.

        Parameters
        ----------
        X : np.array of shape [# startpoints, # simulations per startpoint, dimension]
            data array containing endpoints of trajectory simulations for each startpoint
        """
        super().fit(X)

        self.n_points = X.shape[0] # number of start points
        self.M = X.shape[1] # number of simulations per startpoint
        self.dim = X.shape[2]

        if self.method == "rff":
            self.sampler = RBFSampler(gamma = self.gamma, n_components=self.n_components,
                                      **self.kwargs)
        elif self.method == "nystroem":
            self.sampler = Nystroem(kernel=self.kernel, n_components=self.n_components,
                                    **self.kwargs)
        else:
            raise ValueError("Instantiate with either method='rff' or sampler='nystroem'")

        #compute approximation space mean embeddings
        self.embedded = self.sampler.fit_transform(X.reshape(self.n_points * self.M, self.dim))
        self.embedded = self.embedded.reshape(self.n_points, self.M, self.n_components).sum(axis=1) #/ self.M

        #covariance matrix
        mean = self.embedded.sum(axis=0) / self.n_points
        self.embedded = self.embedded - mean
        cov = self.embedded.T @ self.embedded # n_components x n_components
        _, self.vec = eigsh(cov, k=1, which="LM")

    def predict(self, Y):
        """
        Evaluates the computed eigenfunction on given test data Y.
        Note: fit() has to be run first.

        Parameters
        ----------
        Y : np.array of shape [# testpoints, dimension]
            data array containing endpoints of trajectory simulations for each startpoint
        """
        super().predict(Y)

        Y_embedded = self.sampler.transform(Y)
        return Y_embedded @ self.vec


def _reshape(X):
    """
    This is a temporary auxiliary function to
    bridge the transfer from old twodimensional data interface
    to threedimensional interface.
    It is used to run the old computational routines
    with the new interfaces before updating the computational
    routines for the new interface directly.

    Helper function providing reshape from three dimensional data format
    [# startpoints, # simulations per startpoint, dimension]
    to twodimensional data format
    [# startpoints * # simulations per startpoint, dimension],
    where the first dimension is dominant in # simulations per startpoint
    """
    n_startpoints = X.shape[0]
    M = X.shape[1]
    dim = X.shape[2]

    return X.swapaxes(0,1).reshape(n_startpoints * M, dim)
