
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transition Manifold analysis for the Müller-Brown potential

@author: andreas
"""

# generic imports
import sys
import os

# numerics imports
import numpy as np
import matplotlib.pyplot as plt

# visualization imports
import matplotlib as mpl
from matplotlib import cm

# TM imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import tram.gradient_system as gs
import tram.kernels as krnls
import tram.transition_manifold as tm
import potential


# setting up the system
domain = np.array([[-1.5, -0.5], [1.5, 2.5]])
beta = 1 # inverse energy
potential = potential.Potential()
system = gs.GradientSystem(potential, domain, beta)

# points in which to compute the reaction coordinate
nTestpoints = 32
xtest = system.generateTestpoints(nTestpoints, 'grid')

# points to visualize the potential
nPotpoints = 64
xpot = system.generateTestpoints(nPotpoints, 'grid')


# plotting the Müller-Brown potential
def visualizePotential():
    V = system.pot(xpot)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh( xpot[:,0].reshape(nPotpoints, -1), xpot[:,1].reshape(nPotpoints, -1), V.reshape(nPotpoints,-1), cmap='viridis', vmax=10, shading='gouraud')

    plt.rc('text', usetex=True)
    font = {'family' : 'serif', 
            'size'   : 16}
    plt.rc('font', **font)
    fig.set_size_inches(4,4)
    ax.set_title(r'Mueller-Brown potential')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    
    fig.show()



#######################################
#
#    kernel Transition Manifold method
#
#######################################

# compute Reaction coordinate using the Kernel Embedding of the Transition Manifold
def computeKernelRC():
    # choose the embedding kernel
    kernel = krnls.GaussianKernel(1)
    
    # computation of the reaction corodinate
    kerTM = tm.KernelBurstTransitionManifold(system, kernel, xtest, 0.03, 0.00001, 100, 1)
    kerTM.computeRC()
    return kerTM


# Visualize the reaction coordinate    
def visualizeKernelRC(kerTM):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # visualize RC
    pcol = ax.pcolormesh( xtest[:,0].reshape(nTestpoints, -1),
                 xtest[:,1].reshape(nTestpoints, -1),
                 np.real(kerTM.rc[1][:,1].reshape(nTestpoints,-1)), 
                 shading='flat',
                 )
    clb = fig.colorbar(pcol)
    clb.ax.set_title(r'$\xi$')
    
    plt.rc('text', usetex=True)
    font = {'family' : 'serif', 
            'size'   : 16}
    plt.rc('font', **font)
    fig.set_size_inches(4,4)
    ax.set_title(r'kernel RC')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    
    fig.show()
    
    
##################################################
#
#    Whitney Embedding Transition Manifold method
#
##################################################

def computeEmbeddingRC():
    # embedding function
    embfun=tm.RandomLinearEmbeddingFunction(2,3,0)
    # computation of the reaction corodinate
    embTM=tm.EmbeddingBurstTransitionManifold(system,embfun,xtest,0.03,0.00001,100,.01)
    embTM.computeRC()
    return embTM

    
def visualizeEmbeddingRC(embTM):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # visualize RC
    ax.pcolormesh( xtest[:,0].reshape(nTestpoints, -1),
                 xtest[:,1].reshape(nTestpoints, -1), 
                 np.real(embTM.rc[1][:,1].reshape(nTestpoints,-1)), 
                 shading='flat',
                 )
    
    plt.rc('text', usetex=True)
    font = {'family' : 'serif', 
            'size'   : 16}
    plt.rc('font', **font)
    fig.set_size_inches(4,4)
    ax.set_title(r'embedding RC')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    fig.show()    



def main():
    
    visualizePotential()
    
    print("Computing Reaction coordinate using the Kernel Transition Manifold method")
    kerTM=computeKernelRC()
    visualizeKernelRC(kerTM)
#    
    print("Computing Reaction coordinate using the Whitney Embedding Transition Manifold method")
    embTM=computeEmbeddingRC()
    visualizeEmbeddingRC(embTM)
    
    
if __name__== "__main__":
    main()