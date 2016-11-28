CNN Tutorial
============

This tutorial describes how to implement convolutional neural network (CNN) on MinPy.
CNN is surprisingly effective on computer vision and natural language processing tasks,
it is widely used in real world applications. 

However, these tasks are also extremely computationally demanding. Therefore, training
CNN models effectively calls for GPU acceleration. This tutorial explains how to 
use MinPy's ability to run on GPU transparently for the same model you developped for
CPU. 

This is also a gentle introduction on how to use ``module builder`` to specific an otherwise
complex network. 

We do suggest you start with :ref:`complete_solver_guide` for MinPy's conventional solver architecture.

Dataset: CIFAR-10
-----------------

We use `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset for our CNN model.

CNN on MinPy
------------

In :ref:`complete_solver_guide`, we introduced a simple model/solver architecture.
Implementing CNN in MinPy is straightforward following the convention. The only
difference is the model part. As for the performance critical CNN layers, it is important
to use MXNet symbol, which has been carefully optimized for better performance on GPU. The following
MinPy code defines a classical CNN to classify CIFAR-10 dataset.

If you are running on a server with GPU, **uncommenting line 16** to get the training going
on GPU!

.. literalinclude:: cnn_sym.py
  :language: python
  :linenos:
  
Build Your Network with ``minpy.model_builder``
-----------------------------------------------

``minpy.model_builder`` provides an interface helping you implement models more efficiently.
Model builder generates models compatible with Minpy's solver. You only need to specify basic layer configurations of your model and model builder is going to handle the rest.
Below is a model builder implementation of CNN. Please refer to :ref:`model_builder_tutorial` for details.

**Uncommenting line #20** to train on GPU.

.. literalinclude:: model_builder_example.py
  :language: python
  :linenos:
