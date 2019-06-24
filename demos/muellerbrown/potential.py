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

    def __init__(self, aa = [-1, -1, -6.5, 0.7],
                     bb = [0, 0, 11, 0.6],
                     cc = [-10, -10, -6.5, 0.7],
                     AA = [-200, -100, -170, 15],
                     XX = [1, 0, -0.5, -1],
                     YY = [0, 0.5, 1.5, 1]
                 ):
        self.aa = aa
        self.bb = bb
        self.cc = cc
        self.AA = AA
        self.XX = XX
        self.YY = YY
        """
        Parameters
        ----------
        aa = [-1, -1, -6.5, 0.7],
        bb = [0, 0, 11, 0.6],
        cc = [-10, -10, -6.5, 0.7],
        AA = [-200, -100, -170, 15],
        XX = [1, 0, -0.5, -1],
        YY = [0, 0.5, 1.5, 1]
            parameters of the system
        """


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

        V = 0
        for j in range(4):
            V = V + self.AA[j]*np.exp(self.aa[j]*(x[:, 0] - self.XX[j])**2 + \
                    self.bb[j]*(x[:, 0] - self.XX[j])*(x[:, 1] - self.YY[j]) + \
                    self.cc[j]*(x[:, 1] - self.YY[j])**2)
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

        dVx = 0
        dVy = 0

        for j in range(4):
            ee = self.AA[j]*np.exp(self.aa[j]*(x[:, 0] - self.XX[j])**2 + \
                        self.bb[j]*(x[:, 0] - self.XX[j])*(x[:, 1] - self.YY[j]) + \
                        self.cc[j]*(x[:, 1] - self.YY[j])**2)
            dVx = dVx + (2*self.aa[j]*(x[:, 0] - self.XX[j]) + \
                         self.bb[j]*(x[:, 1] - self.YY[j]))*ee
            dVy = dVy + (self.bb[j]*(x[:, 0] - self.XX[j]) + \
                         2*self.cc[j]*(x[:, 1] - self.YY[j]))*ee

        nablaV = np.transpose(np.array([
                dVx,
                dVy
                ]))
        return nablaV

