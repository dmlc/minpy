Supported GPU operators
=======================

MinPy integrates MXNet operators to enable computation on GPUs. Technically all MXNet GPU operators are supported.

As a reference, following MXNet operators exist.

* Elementwise unary operators

  * Abs
  * Sign
  * Round
  * Ceil
  * Floor
  * Square
  * Square root
  * Exponential
  * Logarithm
  * Cosine
  * Sine

* Elementwise binary operators

  * Plus
  * Minus
  * Multiplication
  * Division
  * Power
  * Maximum
  * Minimum

* Broadcast

  * Norm
  * Maximum
  * Minimum
  * Sum
  * Max axis
  * Min axis
  * Sum axis
  * Argmax channel

* Elementwise binary broadcast

  * Plus
  * Minus
  * Multiplication
  * Division
  * Power

* Matrix

  * Transpose
  * Expand dims
  * Crop
  * Slice axis
  * Flip
  * Dot
  * Batch dot

* Deconvolution
* Sequence mask
* Concatenation
* Cast
* Swap axis
* Block grad
* Leaky relu
* RNN
* Softmax
* Pooling
* Softmax cross entropy
* Sample uniform
* Sample normal
* Smooth L1

But not all MXNet operators are gradable. You can still use them in computation, but trying to ``grad`` them will result in an error.

Following MXNet operators have gradient implemented.

* `Dot <https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html>`_
* `Exponential <https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html>`_
* `Logarithm <https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html>`_
* `Sum <https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html>`_
* `Plus <https://docs.scipy.org/doc/numpy/reference/generated/numpy.add.html>`_
* `Minus <https://docs.scipy.org/doc/numpy/reference/generated/numpy.subtract.html>`_
* `Multiplication <https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html>`_
* `Division <https://docs.scipy.org/doc/numpy/reference/generated/numpy.divide.html>`_
* `True division <https://docs.scipy.org/doc/numpy/reference/generated/numpy.true_divide.html>`_
* `Maximum <https://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html>`_
* `Negation <https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.negative.html>`_
* `Transpose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html>`_
* `Abs <https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html>`_
* `Sign <https://docs.scipy.org/doc/numpy/reference/generated/numpy.sign.html>`_
* `Round <https://docs.scipy.org/doc/numpy/reference/generated/numpy.round_.html>`_
* `Ceil <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ceil.html>`_
* `Floor <https://docs.scipy.org/doc/numpy/reference/generated/numpy.floor.html>`_
* `Sqrt <https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html>`_
* `Sine <https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html>`_
* `Cosine <https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html>`_
* `Power <https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html>`_
* `Reshape <https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html>`_
* `Expand dims <https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html>`_

As for NumPy operators, all preceding operators, plus the following, have gradient defined.

* `Broadcast to <https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html>`_
* `Mod <https://docs.scipy.org/doc/numpy/reference/generated/numpy.mod.html>`_
* `Minimum <https://docs.scipy.org/doc/numpy/reference/generated/numpy.minimum.html>`_
* `Append <https://docs.scipy.org/doc/numpy/reference/generated/numpy.append.html>`_
* Sigmoid
