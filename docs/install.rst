============
Installation
============

Requirements
============

PyTMRC requires Python 3.6.0 or above. Moreover, the following packages are required to make use of all functionality:

- NumPy
- SciPy
- scikit-learn
- Matplotlib
- TQDM

While PyTMRC itself is not currently distributed via a package manager such as `Anaconda <https://www.anaconda.com/>`_ or `PIP <https://pypi.org>`_, its requirements can easily be installed via::

   conda install numpy scipy sklearn matplotlib
   conda install -c conda-forge tqdm

or::

   pip install numpy scipy sklearn matplotlib tqdm


Download and Setup
==================

The latest version of PyTMRC can be obtained from the `Github project page <https://github.com/abittracher/pytmrc>`_::

   git clone https://github.com/abittracher/pytmrc.git

As PyTMRC currently has no automated install routine, we recommend placing the ``tmrc`` folder directly in your source directory, and importing the desired modules via::
   
   import tmrc.system
   import tmrc.kernels
   Import tmrc.manifold_learning
   import tmrc.transition_manifold
   import tmrc.embedding_functions