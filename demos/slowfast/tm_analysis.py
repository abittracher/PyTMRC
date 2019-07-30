
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
import tram.system as system
import tram.kernels as krnls
import tram.transition_manifold as tm
import driftdiffusion


# setting up the system
domain = np.array([[0, 0], [1, 1]])
driftdiffusion = driftdiffusion.DriftDiffusion()
system = system.DriftDiffusionSystem(driftdiffusion, domain)

# points in which to compute the reaction coordinate
nTestpoints = 32
xtest = system.generateTestpoints(nTestpoints, 'grid')


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
    kerTM = tm.KernelBurstTransitionManifold(system, kernel, xtest, 1, 0.001, 100, 1)
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
    embTM=tm.EmbeddingBurstTransitionManifold(system,embfun,xtest,1,0.001,100,.01)
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
    
    
def visualizeTrajectory():

    x0 = np.array([[0.5, 0.5]])
    x = system.generateTrajectory(10,0.001,x0,0)
    
    fig = plt.figure()
    plt.plot(x[:, 0], x[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def visualizeBurst():

    x0 = np.tile(np.array([0.5,0.5]),(1000,1))
    x = system.generateBurst(1,0.001,x0,0)
    
    fig = plt.figure()
    plt.plot(x[:, 0], x[:, 1], 'o')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def visualizePointclouds():

    x0 = np.array([[0.3, 0.5], [0.6, 0.5]])
    X = system.generatePointclouds(1, 0.001, x0, 0, 1000)
    
    fig = plt.figure()
    plt.plot(X[0, :, 0], X[0, :, 1], 'ro')
    plt.plot(X[1, :, 0], X[1, :, 1], 'bo')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



def main():

    visualizePointclouds()
    

    #print("Computing Reaction coordinate using the Kernel Transition Manifold method")
    #kerTM=computeKernelRC()
    #visualizeKernelRC(kerTM)

    #print("Computing Reaction coordinate using the Whitney Embedding Transition Manifold method")
    #embTM=computeEmbeddingRC()
    #visualizeEmbeddingRC(embTM)


if __name__== "__main__":
    main()
