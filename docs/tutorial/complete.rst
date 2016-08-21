Complete solver and optimizer guide
===================================

Feedforward networks
--------------------

In general, we advocate following the common coding style with the following modular partition:

* *Model*: your main job!
* *Layers*: building block of your model.
* *Solver* and *optimizer*: takes your model, training and testing data, train it.

The following MinPy code should be self-explainable; it is a simple two-layer feed-forward network. The model is defined in the ``TwoLayerNet`` class, where the ``init``, ``forward`` and ``loss`` functions specify the parameters to be learnt, how the network computes all the way till the loss, and the computation of the loss itself, respectively. (shall we explain why loss is taken out separtely?). The crucial thing to notice the absense of back-prop, MinPy did it automatically for you.

.. literalinclude:: mlp.py
  :linenos:

This simple network takes several common layers from `layers file <https://github.com/dmlc/minpy/blob/master/minpy/nn/layers.py>`_. The same file contains a few other useful layers, such as batch normalization and dropout. Here is how a new model incorporates them; the only thing changes is the model.

The above code is fully NumPy, and yet it can run on GPU, and without explicit backprop needs. While these features are great, it is by no means complete. For example, it is possible to write nested loops to perform convolution in NumPy, and the code will not break. But the convolution will be ran on CPU only, resulting in horrible runtime performance.

MinPy leverages and integrates seemlessly with MXNet's symbolic programming (see `MXNet Python Symbolic API <https://mxnet.readthedocs.io/en/latest/packages/python/symbol.html>`_). In a nutshell, MXNet's symbolic programming interface allows one to write a sub-DAG with symbolic expression. At runtime, all the computations are transformed into GPU-resided, high-performing kernel.

The following code shows how we replace the first layer with two layers of convolutional kernels, using MXNet. Only the model is shown. You can get ready-to-run code for `convolutional network <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn.py>`_.

::

    class ConvolutionNet(ModelBase):
        def __init__(self,
                     input_size=3 * 32 * 32,
                     hidden_size=512,
                     num_classes=10):
            super(ConvolutionNet, self).__init__()
            # Define symbols that using convolution and max pooling to extract better features
            # from input image.
            net = mx.sym.Variable(name='X')
            net = mx.sym.Reshape(data=net,
                                 shape=(batch_size, 3, 32, 32))
            net = mx.sym.Convolution(data=net,
                                     name='conv',
                                     kernel=(7, 7),
                                     num_filter=32)
            net = mx.sym.Activation(data=net, act_type='relu')
            net = mx.sym.Pooling(data=net,
                                 name='pool',
                                 pool_type='max',
                                 kernel=(2, 2),
                                 stride=(2, 2))
            net = mx.sym.Flatten(data=net)
            # Create forward function and add parameters to this model.
            self.conv = Function(net, input_shapes={'X': (batch_size, input_size)},
                                 name='conv')
            self.add_params(self.conv.get_params())
            # Define ndarray parameters used for classification part.
            output_shape = self.conv.get_one_output_shape()
            conv_out_size = output_shape[1]
            self.add_param(name='w1', shape=(conv_out_size, hidden_size)) \
                .add_param(name='b1', shape=(hidden_size,)) \
                .add_param(name='w2', shape=(hidden_size, num_classes)) \
                .add_param(name='b2', shape=(num_classes,))

        def forward(self, X):
            out = self.conv(X=X, **self.params)
            out = layers.affine(out, self.params['w1'], self.params['b1'])
            out = layers.relu(out)
            out = layers.affine(out, self.params['w2'], self.params['b2'])
            return out

        def loss(self, predict, y):
            return layers.softmax_loss(predict, y)

Of course, in this example, we can program it completely, in fully MXNet symbolic way. You can get the full file `with only MXNet symbols <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn_sym.py>`_. Model is as the followings.

::

    class ConvolutionNet(ModelBase):
        def __init__(self,
                    input_size=3 * 32 * 32,
                    hidden_size=512,
                    num_classes=10):
            super(ConvolutionNet, self).__init__()
            # Define symbols that using convolution and max pooling to extract better features
            # from input image.
            net = mx.sym.Variable(name='X')
            net = mx.sym.Reshape(data=net,
                                shape=(batch_size, 3, 32, 32))
            net = mx.sym.Convolution(data=net,
                                    name='conv',
                                    kernel=(7, 7),
                                    num_filter=32)
            net = mx.sym.Activation(data=net, act_type='relu')
            net = mx.sym.Pooling(data=net,
                                name='pool',
                                pool_type='max',
                                kernel=(2, 2),
                                stride=(2, 2))
            net = mx.sym.Flatten(data=net)
            net = mx.sym.FullyConnected(name='fc1', data=net, num_hidden=hidden_size)
            net = mx.sym.Activation(data=net, act_type='relu')
            # Create forward function and add parameters to this model.
            net = mx.sym.FullyConnected(name='fc2', data=net, num_hidden=num_classes)
            self.cnn = Function(net, input_shapes={'X': (batch_size, input_size)},
                                name='cnn')
            self.add_params(self.cnn.get_params())

        def forward(self, X):
            out = self.cnn(X=X, **self.params)
            return out

        def loss(self, predict, y):
            return layers.softmax_loss(predict, y)

However, the advantage of MinPy is that it brings in additional flexibility when needed, this is especially useful for quick prototyping to validate new ideas. Say we want to add a regularization in the loss term, this is done as the followings. Note that we only changed the ``loss`` function. Full code is available `with regularization <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn_reg.py>`_.

::

    class ConvolutionNet(ModelBase):
        def __init__(self,
                    input_size=3 * 32 * 32,
                    hidden_size=512,
                    num_classes=10):
            super(ConvolutionNet, self).__init__()
            # Define symbols that using convolution and max pooling to extract better features
            # from input image.
            net = mx.sym.Variable(name='X')
            net = mx.sym.Reshape(data=net,
                                shape=(batch_size, 3, 32, 32))
            net = mx.sym.Convolution(data=net,
                                    name='conv',
                                    kernel=(7, 7),
                                    num_filter=32)
            net = mx.sym.Activation(data=net, act_type='relu')
            net = mx.sym.Pooling(data=net,
                                name='pool',
                                pool_type='max',
                                kernel=(2, 2),
                                stride=(2, 2))
            net = mx.sym.Flatten(data=net)
            net = mx.sym.FullyConnected(name='fc1', data=net, num_hidden=hidden_size)
            net = mx.sym.Activation(data=net, act_type='relu')
            net = mx.sym.FullyConnected(name='fc2', data=net, num_hidden=num_classes)
            # Create forward function and add parameters to this model.
            self.cnn = Function(net, input_shapes={'X': (batch_size, input_size)},
                                name='cnn')
            self.add_params(self.cnn.get_params())

        def forward(self, X):
            out = self.cnn(X=X, **self.params)
            return out

        def loss(self, predict, y):
            # Add L2 regularization for all the weights.
            reg_loss = 0.0
            for name, weight in self.params.items():
                reg_loss += np.sum(weight ** 2) * 0.5
            # Compute total loss.
            return layers.softmax_loss(predict, y) + weight_decay * reg_loss

Recurrent networks
------------------

Vanilla RNN

GRU

Put it together: feedforward + recurrent networks
-------------------------------------------------

CS231n image captioning

Reinforcement learning with policy gradient
-------------------------------------------

(overflow texts)

The layer->model->solver (optimizer) pipeline

Flow:
- Description of the general architecture
- Describe the role of the architectural components: layer (operator), model, solver and optimizer
- A diagram of the architecture
- Description of the folders/paths (i.e. where to find the components)

Example 1: two layer fully connected, with batch normalization and drop out (i.e. cs231n); fairly complete code snippets, with link to the full code`

Example 2: adding bn/dropout(?) to show more flexibility

Example 3: replace some of the bottom layers with convolution, using MXNet symbolic computation; link to the code. Keypoint: only show the changes in the model and layer part; keypoint is solver/optimizer need not be touched

Example 4: transition to fully symbolic

Example 5: adding flexibility back by having new op in MinPy

Example 6 and onwards, RNN: image caption, with RNN and convnet (taking the convnet from example 2), show the change in model and layer; link to the complete code

Exmaple 7: RL

Wrap up. Touch also on advanced topics:
- How to deal with multiple losses, possibly attached to different model segments?
- What if different model segments are to use different learning rates?
- ....
