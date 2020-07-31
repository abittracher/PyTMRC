#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various embedding functions used for Whitney embeddings and TM visualization.
"""

# numerics imports
import numpy as np
import scipy.linalg as la

class EmbeddingFunction():
    """
    Parent class for various types of embedding functions.
    """

    def __init__(self, inputdimension, outputdimension):
        """
        Instantiates an embedding function object

        Parameters
        ----------
        inputdimension : int
            Expected dimension of the input points
        outputdimension : int
            Dimension of the output image points
        """
        self.inputdimension = inputdimension
        self.outputdimension = outputdimension

class RandomLinearEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function class for linear embedding functions. The form
    of the function is
    x -> A*x
    where A is a matrix of the correct size with coefficients
    drawn uniformly-randomly in the interval [0,1]
    """

    def __init__(self, inputdimension, outputdimension, seed, orthonormalize = False):
        """
        Instantiates a random linear embedding function object

        Parameters
        ----------
        inputdimension : int
            Expected dimension of the input points
        outputdimension : int
            Dimension of the output image points
        seed : Random seed for coefficient generfunctionation
        orthogonalize : bool
            Toggle to orthonormalize columns of the coefficient matrix
        """
        super().__init__(inputdimension, outputdimension)
        self.seed = seed

        # draw the random coefficients, uniformly 
        np.random.seed(self.seed)
        A = np.random.uniform(0, 1, (self.inputdimension,self.outputdimension))
        
        if orthonormalize: self.A,_ = np.linalg.qr(A,mode='complete')
        self.A = A

    def evaluate(self, x):
        """
        Evaluates embedding function at specified points

        Parameters
        ----------
        x : np.array of shape [# points, inputdimension]
        Array of evaluation points

        Returns
        -------
        np.array of shape [# points, outputdimension]
            array of image points
        """

        y = x.dot(self.A)
        return y
