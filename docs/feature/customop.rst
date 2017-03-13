Customized Operator
===================

Sometimes it is useful to define a customized operator with its own derivatives. Here is a comprehensive example of customized operator: `Example <https://github.com/dmlc/minpy/blob/master/examples/nn/cnn_customop.py>`_.

The following code is taken from our example.

.. literalinclude:: customop_example.py
  :linenos:

As in the example, the forward pass of the operator is defined in a normal python function.
The only discrepancy is the decorator ``@customop('numpy')``.
The decorator will change the function into a class instance with the same name as the function.

The decorator ``customop`` has two options:

* ``@customop('numpy')``: It assumes the arrays in the input and the
  output of the user-defined function are both NumPy arrays.

* ``@customop('mxnet')``: It assumes the arrays in the input and the output of the
  user-defined function are both MXNet NDArrays.

Register derivatives for customized operator
--------------------------------------------

To register a derivative, you first need to define a function that takes output and inputs as
parameters and returns a function, just as the example above. The returned function takes
upstream gradient as input, and outputs downstream gradient. Basically, the returned function
describes how to modify the gradient in the backpropagation process on this specific customized
operator w.r.t. a certain variable.

After derivative function is defined, simply register the function by ``def_grad`` as shown in
the example above. In fact, ``my_softmax.def_grad(my_softmax_grad)`` is the shorthand of
``my_softmax.def_grad(my_softmax_grad, argnum=0)``. Use ``argnum`` to specify which variable to bind
with the given derivative.
