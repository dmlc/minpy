Complete solver and optimizer guide
===================================

This tutorial explains the "pipeline" of a typical research project. The idea is to get a quick prototype with the most flexible and familiar package (NumPy), then move the codebase to a more efficient paradigm (MXNet). Typically, one might need to go back and forth iteratively to refine the model. The choice of performance and flexibility depends on the project stage, and is best left to user to decide. Importantly, we have made it as straightforward as possible. For example:

* There is only one codebase to work with. NumPy and MXNet programming idioms mingle together rather easily.
* Neither style, however, requires the user to explicitly write tedious and (often) error-prone backprop path.
* Switching between GPU and CPU is straightforward, same code runs in either environment with only one line of change.

We will begin with a simple neural network using MinPy/NumPy and its ``Solver`` architecture. Then we will morph it gradually into a fully MXNet implementation, and then add NumPy statements as we see fit. 

We do suggest you start with the simpler logistic regression example `here <https://github.com/dmlc/minpy/blob/master/examples/demo/minpy_tutorial.ipynb>`_.

Stage 0: Setup
---------------
All the codes covered in this tutorial could be found in this `folder <https://github.com/dmlc/minpy/blob/master/examples/nn/>`_. All the codes in this folder are self-contained and ready-to-run. Before running, please make sure that you:

* Correctly install MXNet and MinPy. For guidance, refer to `installation guide <https://minpy.readthedocs.io/en/latest/get-started/install.html>`_.
* Follow the instruction in the `README.md` to download the data.
* Run the example you want.

Stage 1: Pure MinPy
-----------------------
*(This section is also available in iPython Notebook `here <https://github.com/dmlc/minpy/blob/master/examples/nn/tutorials/mlp_nn_basics.ipynb>`_)*

In general, we advocate following the common coding style with the following modular partition:

* *Model*: your main job!
* *Layers*: building block of your model.
* *Solver* and *optimizer*: takes your model, training and testing data, train it.

The following MinPy code should be self-explainable; it is a simple two-layer feed-forward network. The model is defined in the ``TwoLayerNet`` class, where the ``init``, ``forward`` and ``loss`` functions specify the parameters to be learnt, how the network computes all the way till the loss, and the computation of the loss itself, respectively. The crucial thing to note is the **absense of back-prop**, MinPy did it automatically for you.

.. literalinclude:: mlp.py
  :language: python
  :linenos:

This simple network takes several common layers from `layers file <https://github.com/dmlc/minpy/blob/master/minpy/nn/layers.py>`_. The same file contains a few other useful layers, such as batch normalization and dropout. Here is how a new model incorporates them.

.. literalinclude:: mlp_bn_dropout.py
  :language: python
  :emphasize-lines: 14-17, 27-29, 33
  :linenos:

Note that ``running_mean`` and ``running_var`` are defined as auxiliary parameters (``aux_param``). These parameters will not be updated by backpropagation.

The above code looks like *fully NumPy*, and yet it can run on GPU, and without explicit backprop needs. At this point, you might feel a little mysterious of what's happening under the hood. For advanced readers, here are the essential bits:

* The `solver file <https://github.com/dmlc/minpy/blob/master/minpy/nn/solver.py>`_ takes the training and test dataset, and fits the model.
* At the end of the ``_step`` function, the ``loss`` function is auto-differentiated, deriving gradients to update the parameters.


Stage 2: MinPy + MXNet
-----------------------

While these features are great, it is by no means complete. For example, it is possible to write nested loops to perform convolution in NumPy, and the code will not break. However, much better implementations exist, especially when running on GPU.

MinPy leverages and integrates seemlessly with MXNet's symbolic programming (see `MXNet Python Symbolic API <https://mxnet.readthedocs.io/en/latest/packages/python/symbol.html>`_). In a nutshell, MXNet's symbolic programming interface allows one to write a sub-DAG with symbolic expression. MXNet's convolutional kernel runs on both CPU and GPU, and its GPU version is highly optimized. 

The following code shows how we add one convolutional layer and one pooling layer, using MXNet. Only the model is shown. You can get ready-to-run code for `convolutional network <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn.py>`_.

.. literalinclude:: cnn.py
  :language: python
  :emphasize-lines: 14-27,37
  :linenos:

Stage 3: Pure MXNet
-------------------

Of course, in this example, we can program it in fully MXNet symbolic way. You can get the full file `with only MXNet symbols <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn_sym.py>`_. Model is as the followings.

.. literalinclude:: cnn_sym.py
  :language: python
  :emphasize-lines: 15-27,30,34
  :linenos:

Stage 3: MXNet + MinPy
---------------------

However, the advantage of MinPy is that it brings in additional flexibility when needed, this is especially useful for quick prototyping to validate new ideas. Say we want to add a regularization in the loss term, this is done as the followings. Note that we only changed the ``loss`` function. Full code is available `with regularization <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn_reg.py>`_.

.. literalinclude:: cnn_reg.py
  :language: python
  :linenos:

Recurrent networks
------------------

The flexibility of MinPy makes it easy to implement other models. The full code is `here <https://github.com/dmlc/minpy/blob/master/examples/nn/rnn.py>`_.

Put it together: feedforward + recurrent networks
-------------------------------------------------

Coming soon (we may not use CS231n image captioning as it leaks homework solution?)

Reinforcement learning with policy gradient
-------------------------------------------

Coming soon.
