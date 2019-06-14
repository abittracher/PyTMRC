#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Müller-Brown potential and its gradient
"""

import math
import numpy as np

   
    
class Potential:
    
    def __init__(self):
        self.aa = [-1, -1, -6.5, 0.7]
        self.bb = [0, 0, 11, 0.6]
        self.cc = [-10, -10, -6.5, 0.7]
        self.AA = [-200, -100, -170, 15]
        self.XX = [1, 0, -0.5, -1]
        self.YY = [0, 0.5, 1.5, 1]
        
    def pot(self, x):
    # potential energy function of the Müller-Brown potential
        V = 0
        for j in range(4):
            V = V + self.AA[j]*np.exp(self.aa[j]*(x[:,0] - self.XX[j])**2 + self.bb[j]*(x[:,0] - self.XX[j])*(x[:,1] - self.YY[j]) + self.cc[j]*(x[:,1] - self.YY[j])**2);
        return V
    
    def gradPot(self, x):
    # gradient of the potential energy function of the Müller-Brown potential
        dVx = 0
        dVy = 0
        
        for j in range(4):
            ee = self.AA[j]*np.exp(self.aa[j]*(x[:,0]-self.XX[j])**2+self.bb[j]*(x[:,0]-self.XX[j])*(x[:,1]-self.YY[j])+self.cc[j]*(x[:,1]-self.YY[j])**2)
            dVx = dVx + (2*self.aa[j]*(x[:,0]-self.XX[j])+self.bb[j]*(x[:,1]-self.YY[j]))*ee;
            dVy = dVy + (self.bb[j]*(x[:,0]-self.XX[j])+2*self.cc[j]*(x[:,1]-self.YY[j]))*ee;
            
        
        nablaV = np.transpose(np.array([
                dVx,
                dVy
                ]))
        return nablaV

