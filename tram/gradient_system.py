#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# numerics imports
import math
import numpy as np
import progressbar


class GradientSystem(object):
    """
    The gradient system class

    Describes dynamical systems whose movement is defined by a gradient drift
    (i.e. pointing in the negative direction of the gradient of a potential
    energy function) and temperature-scaled random fluctuation.

    Attributes
    ----------
    domain : np.array
        an array containing the limits of the system's domain
    dimension : int
        dimension of the system
    beta : float
        inverse temperature

    Methods
    -------
    potential(np.array)
        evaluates the potential energy function
    gradPot(np.array)
        evaluates the gradient of the potential energy function
    computeTrajectory(t, dt, r0)
        computes a single trajectory (including the steps) using the
        Euler-Maruyama scheme
    computeBurst(t, dt, r0 showprogress=True)
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

    def computeTrajectory(self, t, dt, r0, showprogress=True):
        """
        computes a single trajectory (including the steps) using the
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

        if showprogress:
            ProgressBar = progressbar.ProgressBar
        else:
            ProgressBar = progressbar.NullBar
        bar = ProgressBar()

        nsteps = math.floor(t/dt)
        # only first entry of r0 is used as starting point. all other entries are ignored
        rnew = np.array([r0[0, :]])
        sysdim = np.size(rnew)
        rout = np.empty((nsteps+1, sysdim))
        rout[0, :] = rnew

        # parameters for Brownian motion
        mean = np.zeros(sysdim)
        var = (dt*2/self.beta)*np.eye(sysdim)

        for i in bar(range(nsteps)):
            nablaV = self.gradPot(rnew)

            # generate Brownian motion
            dW = np.random.multivariate_normal(mean, var)

            # update position
            rnew = rnew - dt*nablaV + dW
            rout[i, :] = rnew
        return rout

    def computeBurst(self, t, dt, r0, showprogress=True):
        """
        computes in parallel multiple trajectories (outputting only the end points)
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

        if showprogress:
            ProgressBar = progressbar.ProgressBar
        else:
            ProgressBar = progressbar.NullBar
        bar = ProgressBar()

        nsteps = math.floor(t/dt)
        npoints = np.size(r0, 0)
        sysdim = np.size(r0, 1)

        # Brownian motion parameters
        mean = np.zeros(sysdim)
        var = (dt*2/self.beta)*np.eye(sysdim)

        # pre-draw the Brownian motion
        dW = np.random.multivariate_normal(mean, var, nsteps*npoints).reshape((npoints,sysdim,-1))

        print("Burst integration...")
        for i in bar(range(nsteps)):
            nablaV = self.gradPot(r0)

            # draw Brownian motion
            # dW = np.random.multivariate_normal(mean, var, npoints)

            # update position
            #r0 = r0 - dt*nablaV + dW
            r0 = r0 - dt*nablaV + dW[:, :, i]
        return r0

    def generateTestpoints(self, n, dist='uniform'):
        """
        samples the domain of the system

        Parameters
        ----------
        n : int
            number of points to generate
        dist='uniform' : string
            distribution of the points

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
