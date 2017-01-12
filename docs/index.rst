Welcome to MinPy's documentation!
=================================

MinPy aims at prototyping a pure `NumPy <http://www.numpy.org/>`_ interface above `MXNet <https://github.com/dmlc/mxnet>`_ backend. This package targets two groups of users: 

* The beginners who wish to have a firm grasp of the fundamental concepts of deep learning, and 
* Researchers who want a quick prototype of advanced and complex algorithms. 

It is not intended for those who want to compose with ready-made components, although there are enough (layers and activation functions etc.) to get started.

As much as possible, MinPy strikes to be as much as possible to be purely NumPy-compatible. It also abides to a fully imperative programming experience that is familiar to most users. Letting go the popular approach that mixes in symbolic programming sacrifices some runtime optimization opportunities, in favor of algorithmic expressiveness and flexibility. However, its performance is reasonably close to other state-of-art systems, especially when computation dominates.  

This document describes its main features: 

* Auto-differentiation 
* Transparent CPU/GPU acceleration
* Visualization using TensorBoard
* Learning deep learning using MinPy

.. toctree::
    :maxdepth: 2
    :caption: Basics
    :glob:

    get-started/install.rst
    get-started/logistic_regression.rst
    tutorial/numpy_under_minpy.ipynb
    tutorial/autograd_tutorial.ipynb
    tutorial/transparent_fallback.ipynb

.. toctree::
    :maxdepth: 2
    :caption: Advanced Tutorials
    :glob:

    tutorial/complete_sol_opt_guide/complete.rst
    tutorial/cnn_tutorial/cnn_tutorial.rst
    tutorial/rnn_tutorial/rnn_tutorial.rst
    tutorial/rnn_mnist.ipynb
    tutorial/rl_policy_gradient_tutorial/rl_policy_gradient.rst
    tutorial/model_builder_tutorial/model_builder.rst
    tutorial/dl_learn.rst

.. toctree::
    :maxdepth: 2
    :caption: Features
    :glob:

    feature/policy.rst
    feature/op_stat.rst
    feature/context.rst
    feature/*

.. toctree::
    :maxdepth: 2
    :caption: Visualization
    :glob:
    
    tutorial/visualization_tutorial/minpy_visualization.ipynb
    tutorial/visualization_tutorial/minpy_visualization_adv.ipynb

.. toctree::
    :maxdepth: 2
    :caption: Developer Documentation
    :glob:

    how-to/*
    api/modules.rst

.. toctree:: 
    :maxdepth: 2
    :caption: History and Acknowledgement
    :glob:
    
    misc/credits.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

