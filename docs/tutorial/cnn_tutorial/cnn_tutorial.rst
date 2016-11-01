CNN Tutorial
============

This tutorial describes how to implement convolutional neural network (CNN) on MinPy.
CNN is surprisingly effective on computer vision tasks and is widely used in real world
applications. However, vision related tasks takes images as input, which usually contain
large number of pixels. This results in huge amount of computation on the network. In
order to train CNN models effectively, GPU acceleration is more than necessary. In this
tutorial, we will show you how MinPy utilize GPU to acceleration CNN models.

We do suggest you start with :ref:`complete_solver_guide` for MinPy's
conventional solver architecture.

Dataset: CIFAR-10
-----------------

We use `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset for our CNN model.

CNN on MinPy
------------

In :ref:`complete_solver_guide`, we introduced a simple model/solver architecture.
Implementing CNN in MinPy is very straightforward following the convention. The only
difference is the model part. As for the performance critical CNN layers, it is important
to use MXNet symbol, since it is carefully optimized for better performance on GPU. The following
MinPy code defines a classical CNN to classify CIFAR-10 dataset.

.. literalinclude:: cnn_sym.py
  :language: python
  :linenos: