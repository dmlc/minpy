Complete solver and optimizer guide
===================================

----
Feedforward networks
----

In general, we advocate following the common coding style with the following modular partition:

* *Model*: your main job!
* *Layers*: building block of your model.
* *Solver* and *optimizer*: takes your model, training and testing data, train it.

The following Minpy code should be self-explainable; it is a simple two-layer feed-forward network. The model is defined in the ``TwoLayerNet`` class, where the ``init``, ``forward`` and ``loss`` functions specify the parameters to be learnt, how the network computes all the way till the loss, and the computation of the loss itself, respectively. (shall we explain why loss is taken out separtely?). The crucial thing to notice the absense of back-prop, Minpy did it automatically for you.

.. literalinclude:: mlp.py
  :linenos:

This simple network takes several common layers from XX. The same folder contains a few other useful layers, such as batch normalization and dropout. Here is how a new model incorporates them; the only thing changes is the model.

XXX a few lines of the new model.

The above code is fully numpy, and yet it can run on GPU, and without explicit backprop needs. While these features are great, it is by no means complete. For example, it is possible to write nested loops to perform convolution in numpy, and the code will not break. But the convolution will be ran on CPU only, resulting in horrible runtime performance.

Minpy leverages and integrates seemlessly with MXNet's symbolic programming (see HERE). In a nutshell, MXNet's symbolic programming interface allows one to write a sub-DAG with symbolic expression. At runtime, all the computations are transformed into GPU-resided, high-performing kernel.

The following code chance shows how we replace the first layer with two layers of convolutional kernels, using MXNet.

Of course, in this example, we can program it completely, in fully MXNet symbolic way. As the followings:
(code snippet here)

However, the advantage of Minpy is that it brings in additional flexibility when needed, this is especially useful for quick prototyping to validate new ideas. Say we want to add a regularization in the loss term, this is done as the followings.

(code snippet here)


----
Recurrent networks
----

Vanilla RNN

GRU

---------
Put it together: feedforward + recurrent networks
---------

CS231n image captioning

---------
Reinforcement learning with policy gradient
---------

===============================
(overflow texts)

The layer->model->solver (optimizer) pipeline

Flow: 
- Description of the general architecture
- Describe the role of the architectural components: layer (operator), model, solver and optimizer
- A diagram of the architecture
- Description of the folders/paths (i.e. where to find the components)

Example 1: two layer fully connected, with batch normalization and drop out (i.e. cs231n); fairly complete code snippets, with link to the full code`

Example 2: adding bn/dropout(?) to show more flexibility

Example 3: replace some of the bottom layers with convolution, using mxnet symbolic computation; link to the code. Keypoint: only show the changes in the model and layer part; keypoint is solver/optimizer need not be touched

Example 4: transition to fully symbolic

Example 5: adding flexibility back by having new op in minpy

Example 6 and onwards, RNN: image caption, with RNN and convnet (taking the convnet from example 2), show the change in model and layer; link to the complete code

Exmaple 7: RL

Wrap up. Touch also on advanced topics:
- How to deal with multiple losses, possibly attached to different model segments?
- What if different model segments are to use different learning rates?
- ....
