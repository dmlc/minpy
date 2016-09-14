Complete model builder guide
============================

This tutorial demonstrates how to use Minpy's model builder to construct neural networks and train the networks with Minpy's solver. Minpy's model builder provides an interface that simplifies the syntax of network declaration. Networks constructed by model builder is compatible with Minpy's solver, which enables the networks to be trained directly by solver as described `here <https://github.com/dmlc/minpy/blob/master/examples/demo/minpy_tutorial.ipynb>`_.

It is recommended to read `Minpy's solver tutorial <https://github.com/dmlc/minpy/blob/master/examples/demo/minpy_tutorial.ipynb>`_ to be familiarized with basic solver usage.

Get started
-----------
Minpy's model builder consists of classes describing various neural network architectures. ``Sequence`` class enables users to create feedforward networks. There are other classes representing layers of neural networks, such as ``Affine`` and ``Convolution``. `This <https://github.com/dmlc/minpy/blob/master/examples/nn/model_builder.py>`_ demonstrates the declaration and training of a 2-layer fully connected network with classes in model builder. In fact, arbitrarily complex networks could be constructed by combining these classes in a nested way. Please refer to the `model gallery <https://github.com/dmlc/minpy/blob/master/examples/nn/model_gallery.py>`_ to discover how to easily declare giant networks such as ResNet with model builder.

Customize model builder layers
------------------------------
Minpy's model builder is designed to be extended easily. To create customized layers, one only needs to inherit classes from ``minpy.nn.model_builder.Module`` class and implement several inherited functions. These functions are ``forward``, ``output_shape``, ``parameter_shape`` and ``parameter_settings``. ``forward`` receives input data and a dictionary containing the parameters of the network, and generates output data. It could be implemented by Numpy syntax, layers provided in ``minpy.nn.layers``, MXNet symbols or their combination as described in `Minpy's solver tutorial <https://github.com/dmlc/minpy/blob/master/examples/demo/minpy_tutorial.ipynb>`_. ``output_shape`` returns the layer's output shape given the input shape. Please be aware that the shapes should be tuples specifying shape of one training sample, i.e. the shape of CIFAR-10 data is either (3072,) or (3, 32, 32). ``parameter_shape`` returns a dictionary that includes the shapes of all parameters of the layer. One could pass more information to Minpy's solver in ``parameter_settings``. It is optional to implement ``parameter_settings``. The `model builder script <https://github.com/dmlc/minpy/blob/master/minpy/nn/model_builder.py>`_ should be self-explanatory.
