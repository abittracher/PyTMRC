#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Müller-Brown potential

A two-dimensional potential with three metastable sets that is commonly-used as a
benchmark system.
"""

# numerics imports
import numpy as np


class Potential:
    
    def __init__(self, ystretch = 10):
        self.ystretch = ystretch

    def pot(self, x):
        """
        The potential energy function

        Parameters
        ----------
        x : np.array
            array containing the evaluation points

        Output
        ------
        V : np.array
            potential evaluated at the evaluation points
        """

        V = self.ystretch*x[:, 1]**2;
        
        return V

    def gradPot(self, x):
        """
        The gradient of the potential energy function

        Parameters
        ----------
        x : np.array
            array containing the evaluation points

        Output
        ------
        nablaV : np.array
            gradient evaluated at the evaluation points
        """

        dVx = 0 * x[:, 0]
        dVy = 2 * self.ystretch*x[:,1]

        nablaV = np.transpose(np.array([
                dVx,
                dVy
                ]))
        return nablaV

