PyTMRC
******

What is PyTMRC?
===============

PyTMRC is the Python Transition Manifold Reaction Coordinate package, a software package for computing reaction coordinates of stochastic dynamical systems based on the transition manifold data analysis framework. Its main application is the analysis of large atomistic models from molecular dynamics, but it can be applied to multi-scale processes from other areas of science as well.

The latest version of the software is available on `github <http://github.com/abittracher/pytmrc>`_. A build of the documentation can be found at Read the docs.

Features
========

* Data-driven computation and visualisation of reaction coordinates
* Implements the base transition manifold method and all recent extensions
* compatible with time series-type as well as parallel burst-type sampling data
* easy to use, scikit learn-inspired interface
* efficient numerics through NumPy backend

Publications
============

The methods and mathematical framework implemented by PyTMRC are described in detail in the following publications:

* Bittracher, Koltai, Klus, Banisch, Dellnitz, Schütte, “Transition Manifolds of Complex Metastable Systems: Theory and Data-Driven Computation of Effective Dynamics,” J. Nonlinear Sci. 28, no. 2 (2017): 471–512, https://doi.org/10.1007/s00332-017-9415-0.
* Bittracher, Banisch, Schütte, “Data-Driven Computation of Molecular Reaction Coordinates,” The Journal of Chemical Physics 149, no. 15 (2018): 154103, https://doi.org/10.1063/1.5035183.
* Bittracher, Klus, Hamzi, Koltai, Schütte, “Dimensionality Reduction of Complex Metastable Systems via Kernel Embeddings of Transition Manifolds,” To Appear in Journal of Nonlinear Science, 2019, https://arxiv.org/abs/1904.08622.


Contributors
============

The package was created by Andreas Bittracher and is developed and maintained by Andreas Bittracher and Mattes Mollenhauer. During the time of the development, Andreas Bittracher was funded by Deutsche Forschungsgemeinschaft (DFG) through grant CRC 1114 “Scaling Cascades in Complex Systems”, Project Number 235221301, Project B03 “Multi- level coarse graining of multiscale problems”, and Mattes Mollenhauer was funded by by Deutsche Forschungsgemeinschaft (DFG) through grant EXC 2046 “MATH+”, Project Number 390685689, Project AA1-2 “Learning Transition Manifolds and Effective Dynamics of Biomolecules”.


Users
=====

The software is used by researchers of the Collaborative Research Center 1114 centered at Freie Universität Berlin, the Math+ Berlin Mathematics Research Center, as well as the Zuse Institute Berlin.


License
=======

BSD 3-Clause License. See the LICENSE file for details.
