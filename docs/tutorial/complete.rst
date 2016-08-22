Complete solver and optimizer guide
===================================

Feedforward networks
--------------------

In general, we advocate following the common coding style with the following modular partition:

* *Model*: your main job!
* *Layers*: building block of your model.
* *Solver* and *optimizer*: takes your model, training and testing data, train it.

The following MinPy code should be self-explainable; it is a simple two-layer feed-forward network. The model is defined in the ``TwoLayerNet`` class, where the ``init``, ``forward`` and ``loss`` functions specify the parameters to be learnt, how the network computes all the way till the loss, and the computation of the loss itself, respectively. The crucial thing to note is the absense of back-prop, MinPy did it automatically for you.

.. literalinclude:: mlp.py
  :linenos:

This simple network takes several common layers from `layers file <https://github.com/dmlc/minpy/blob/master/minpy/nn/layers.py>`_. The same file contains a few other useful layers, such as batch normalization and dropout. Here is how a new model incorporates them.

(code with BN and dropout, dropout only at the very last layer, BN can be anywhere)

The above code is fully NumPy, and yet it can run on GPU, and without explicit backprop needs. At this point, you might feel a little mysterious of what's happening under the hood. For advanced readers, here are the essential bits:

* The `solver file <https://github.com/dmlc/minpy/blob/master/minpy/nn/solver.py>`_ takes the training and test dataset, and fits the model.
* At the end of the ``_step`` function, the ``loss`` function is auto-differentiated, deriving gradients to update the parameters.

While these features are great, it is by no means complete. For example, it is possible to write nested loops to perform convolution in NumPy, and the code will not break. But the convolution will be ran on CPU only, resulting in horrible runtime performance.

MinPy leverages and integrates seemlessly with MXNet's symbolic programming (see `MXNet Python Symbolic API <https://mxnet.readthedocs.io/en/latest/packages/python/symbol.html>`_). In a nutshell, MXNet's symbolic programming interface allows one to write a sub-DAG with symbolic expression. MXNet's convolutional kernel runs on both CPU and GPU, and its GPU version is highly optimized. 

The following code shows how we replace the first layer with two layers of convolutional kernels, using MXNet. Only the model is shown. You can get ready-to-run code for `convolutional network <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn.py>`_.

.. literalinclude:: cnn.py
  :linenos:

Of course, in this example, we can program it completely, in fully MXNet symbolic way. You can get the full file `with only MXNet symbols <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn_sym.py>`_. Model is as the followings.

.. literalinclude:: cnn_sym.py
  :linenos:

However, the advantage of MinPy is that it brings in additional flexibility when needed, this is especially useful for quick prototyping to validate new ideas. Say we want to add a regularization in the loss term, this is done as the followings. Note that we only changed the ``loss`` function. Full code is available `with regularization <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn_reg.py>`_.

.. literalinclude:: cnn_reg.py
  :linenos:

Recurrent networks
------------------

Coming soon (Larry)

Put it together: feedforward + recurrent networks
-------------------------------------------------

Coming soon (we may not use CS231n image captioning as it leaks homework solution?)

Reinforcement learning with policy gradient
-------------------------------------------

Coming soon.
