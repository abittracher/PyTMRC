#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# system imports
import sys

# numerics imports
import math
import numpy as np

# utility imports
from tqdm import tqdm
import time

class System:
    """
    Parent class for defining stochastic dynamical systems.

    Only initializes some basic properties of a dynamical system. Other important
    properties depend on the specific kind of system.
    
    Attributes
    ----------
    domain : np.array
        an array containing the limits of the system's domain
    dimension : int
        dimension of the system's state space

    Methods
    -------
    generateTestPoints(n, dist='uniform')
        samples the domain of the system
    """

    def __init__(self, domain):
        """
        Parameters
        ----------
        domain : np.array
            an array containing the limits of the system's domain
        beta : float
            inverse temperature
        """

        self.domain = domain
        self.dimension = np.size(domain, 1)
    

    def generateTestpoints(self, n, dist='uniform'):
        """
        Generates samples of the system's state space

        Parameters
        ----------
        n : int
            number of points to generate
        dist='uniform' : string
            method of distribution the points. Can be either 'uniform' or 'grid'

        Output
        ------
        rsamp : np.array
            array containing the generated sampling points
        """

        # uniformly-random distribution
        if dist == 'uniform':
            rsamp = np.random.uniform(self.domain[0], self.domain[1], (n, self.dimension))
            return rsamp
        # regular grid distribution
        elif dist == 'grid':
            Xls = []
            for i in range(self.dimension):
                Xls.append(np.linspace(self.domain[0][i], self.domain[1][i], n))
            X = np.meshgrid(*Xls)
            X = np.asarray(X)
            rsamp = np.vstack([X[0].ravel(), X[1].ravel()]).transpose()
            return rsamp


class GradientSystem(System):
    """
    The gradient system class

    Implements a dynamical system whose movement is defined by a gradient drift
    (i.e. pointing in the negative direction of the gradient of a potential
    energy function) and temperature-scaled isotropic noise.

    Attributes
    ----------
    domain : np.array
        an array containing the limits of the system's domain
    dimension : int
        dimension of the system's state space
    beta : float
        inverse temperature

    Methods
    -------
    potential(np.array)
        evaluates the potential energy function
    gradPot(np.array)
        evaluates the gradient of the potential energy function
    generateTrajectory(t, dt, r0)
        computes a single trajectory (including the steps) using the
        Euler-Maruyama scheme
    generateBurst(t, dt, r0, showprogress=True)
        computes multiple trajectories (outputting only the end points)
        using the Euler-Maruyama scheme
    generateTestPoints(n, dist='uniform')
        samples the domain of the system
    """

    def __init__(self, potential, domain, beta):
        """
        Parameters
        ----------
        potential : potential
            object containing information about the potential energy function
            and its gradient
        domain : np.array
            an array containing the limits of the system's domain
        beta : float
            inverse temperature
        """

        self.pot = potential.pot
        self.gradPot = potential.gradPot
        self.domain = domain
        self.dimension = np.size(domain, 1)
        self.beta = beta

    def generateTrajectory(self, t, dt, r0, showprogress=True):
        """
        Computes a single trajectory (including the steps) using the
        Euler-Maruyama integration scheme.

        Parameters
        ----------
        t : float
            end time. Method integrates the system from 0 to t
        dt : float
            time step length of the integration
        r0 : np.array
            starting point
        showprogress=True : bool
            flag for progress bar

        Output
        ------
        rout : np.array
            array containing the steps of the integration

        """

        nsteps = math.floor(t/dt)
        # only first entry of r0 is used as starting point. all other entries are ignored
        rnew = np.array([r0[0, :]])
        sysdim = np.size(rnew)
        rout = np.empty((nsteps, sysdim))
        rout[0, :] = rnew

        # parameters for Brownian motion
        mean = np.zeros(sysdim)
        var = (dt*2/self.beta)*np.eye(sysdim)

        if showprogress: 
            print("Generating trajectory...")
            sys.stdout.flush() # workaround for messed-up progress bars
            
        for i in tqdm(range(nsteps), disable=not showprogress):
            nablaV = self.gradPot(rnew)

            # generate Brownian motion
            dW = np.random.multivariate_normal(mean, var)

            # update position
            rnew = rnew - dt*nablaV + dW
            rout[i, :] = rnew
        return rout

    def generateBurst(self, t, dt, r0, showprogress=True):
        """
        Computes in parallel multiple trajectories (outputting only the end points)
        using the Euler-Maruyama integration scheme.

        Parameters
        ----------
        t : float
            end time. Method integrates the system from 0 to t
        dt : float
            time step length of the integration
        r0 : np.array
            array containing the starting points
        showprogress=True : bool
            flag for progress bar

        Output
        ------
        rout : np.array
            array containing the end points of the trajectories

        """
        
    
        nsteps = math.floor(t/dt)
        npoints = np.size(r0, 0)
        sysdim = np.size(r0, 1)

        # Brownian motion parameters
        mean = np.zeros(sysdim)
        var = (dt*2/self.beta)*np.eye(sysdim)

        # pre-draw the Brownian motion
        #dW = np.random.multivariate_normal(mean, var, nsteps*npoints).reshape((npoints,sysdim,-1))
        
        if showprogress: 
            print("Burst integration...")
            sys.stdout.flush() # workaround for messed-up progress bars

        for i in tqdm(range(nsteps), disable = not showprogress):
            nablaV = self.gradPot(r0)

            # draw Brownian motion
            dW = np.random.multivariate_normal(mean, var, npoints)

            # update position
            r0 = r0 - dt*nablaV + dW
            #r0 = r0 - dt*nablaV + dW[:, :, i]
        return r0

    def generatePointclouds(self, t, dt, r0, M, showprogress=True):
        """
        Computes the endpoints of multiple parallel trajectories for each starting point 
        in the array r0 using the Euler-Maruyama integration scheme. Implemented as
        wrapper for generateBurst.

        Parameters
        ----------
        t : float
            end time. Method integrates the system from 0 to t
        dt : float
            time step length of the integration
        r0 : np.array
            array containing the starting points
        M : int
            number of simulations per starting point
        showprogress=True : bool
            flag for progress bar

        Output
        ------
        pointclouds : np.array
            3D array containing pointcloud data (npoints x M x dim-array)
        """
                
        npoints = np.size(r0, 0)
        r1 = np.zeros((npoints, M, self.dimension))

        if showprogress: 
            tqdm.write("Generating poinclouds...")
            sys.stdout.flush() # workaround for messed-up progress bars

        for k in tqdm(range(npoints)):
            r0M = np.tile(r0[k],(M,1))
            r1M = self.generateBurst(t, dt, r0M, showprogress=False)
            r1[k] = r1M

        return r1


class DriftDiffusionSystem(System):
    """
    The drift diffusion system class

    Implements a dynamical system whose movement is defined by generic drift and diffuion
    terms. Both drift and diffusion may be location and time-dependent.

    Attributes
    ----------
    domain : np.array
        an array containing the limits of the system's domain
    dimension : int
        dimension of the system's state space

    Methods
    -------
    drift(np.array)
        evaluates the drift term
    diffusion(np.array)
        evaluates the diffusion term
    generateTrajectory(t, dt, r0)
        computes a single trajectory (including the steps) using the
        Euler-Maruyama scheme
    generateBurst(t, dt, r0, showprogress=True)
        computes multiple trajectories (outputting only the end points)
        using the Euler-Maruyama scheme
    generateTestPoints(n, dist='uniform')
        samples the domain of the system
    """

    def __init__(self, driftdiffusion, domain):
        """
        Parameters
        ----------
        driftdiffusion : driftdiffusion
            object containing the drift and diffusion of the system
        domain : np.array
            an array containing the limits of the system's domain
        """

        self.drift = driftdiffusion.drift
        self.diffusion = driftdiffusion.diffusion
        self.domain = domain
        self.dimension = np.size(domain, 1)

    def generateTrajectory(self, t, dt, r0, t0, showprogress=True):
        """    
        Computes a single trajectory (including the steps) using the
        Euler-Maruyama integration scheme.

        Parameters
        ----------
        t : float
            end time. Method integrates the system from 0 to t
        dt : float
            time step length of the integration
        r0 : np.array
            starting point
        t0 : float
            starting time
        showprogress=True : bool
            flag for progress bar

        Output
        ------
        rout : np.array
            array containing the steps of the integration

        """

        try: 
            assert t > t0, "End time must be larger than start time."
        except AssertionError as e:
            print(e)

        nsteps = math.floor((t-t0)/dt)    
        rnew = np.array([r0[0, :]]) # only first entry of r0 is used as starting point. all other entries are ignored
        tnew = t0
        sysdim = np.size(rnew)
        rout = np.empty((nsteps, sysdim))
        rout[0, :] = rnew

        # parameters for Brownian motion, as required by np.random.multivariate_normal
        mean = np.zeros(sysdim)
        var = dt*np.eye(sysdim)

        if showprogress: 
            print("Generating trajectory...")
            sys.stdout.flush() # workaround for messed-up progress bars

        for i in tqdm(range(nsteps), disable=not showprogress):
        
            b = self.drift(tnew, rnew) # evaluate drift
            sigma = self.diffusion(tnew, rnew) # evaluate diffusion matrix

            # generate Brownian motion
            dW = np.random.multivariate_normal(mean, var)

            # update position
            tnew = tnew + dt
            rnew = rnew + dt*b + sigma @ dW
            rout[i, :] = rnew
        return rout

    def generateBurst(self, t, dt, r0, t0, showprogress=True):
        """
        Computes in parallel multiple trajectories (outputting only the end points)
        using the Euler-Maruyama integration scheme.

        Parameters
        ----------
        t : float
            end time. Method integrates the system from 0 to t
        dt : float
            time step length of the integration
        r0 : np.array
            array containing the starting points
        t0 : float
            start time
        showprogress=True : bool
            flag for progress bar

        Output
        ------
        rout : np.array
            array containing the end points of the trajectories

        """
        try: 
            assert t > t0, "End time must be larger than start time."
        except AssertionError as e:
            print(e)
    
        nsteps = math.floor((t-t0)/dt)
        npoints = np.size(r0, 0)
        sysdim = np.size(r0, 1)

        # Brownian motion parameters, as required by np.random.multivariate_normal
        mean = np.zeros(sysdim)
        var = dt*np.eye(sysdim)
        
        if showprogress: 
            print("Burst integration...")
            sys.stdout.flush() # workaround for messed-up progress bars

        for i in tqdm(range(nsteps), disable = not showprogress):
            b = self.drift(t0, r0) # evaluate drift
            sigma = self.diffusion(t0, r0) # evaluate diffusion

            # draw Brownian motion
            dW = np.random.multivariate_normal(mean, var, npoints)

            # update time and position
            t0 = t0 + dt
            r0 = r0 + dt*b + np.einsum('...kl,...l', sigma, dW)
        return r0

    
    def generatePointclouds(self, t, dt, r0, t0, M, showprogress=True):
        """
        Computes the endpoints of multiple parallel trajectories for each starting point 
        in the array r0 using the Euler-Maruyama integration scheme. Implemented as
        wrapper for generateBurst.

        Parameters
        ----------
        t : float
            end time. Method integrates the system from 0 to t
        dt : float
            time step length of the integration
        r0 : np.array
            array containing the starting points
        t0 : float
            starting time
        M : int
            number of simulations per starting point
        showprogress=True : bool
            flag for progress bar

        Output
        ------
        pointclouds : np.array
            3D array containing pointcloud data (npoints x M x dim-array)
        """
                
        npoints = np.size(r0, 0)
        r1 = np.zeros((npoints, M, self.dimension))

        if showprogress: 
            print("Generating poinclouds...")
            sys.stdout.flush() # workaround for messed-up progress bars

        for k in tqdm(range(npoints)):
            r0M = np.tile(r0[k],(M,1))
            r1M = self.generateBurst(t, dt, r0M, t0, showprogress=False)
            r1[k] = r1M

        return r1