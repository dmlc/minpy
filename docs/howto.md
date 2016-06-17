# MinPy guide

This is a series of guide on using various parts of MinPy.

## NumPy compatability

## Auto differentiation

## Hybrid programming paradigm

### Using MinPy's internal operators

Since MinPy inherits nearly all operators from NumPy, you could expect to use
a NumPy operator without additional instructions. By nature of the MinPy operator
with the same name as the NumPy equivalent, it might actually run the NumPy
implementation or corresponding MXNet implementation, depending on the policy.

Apart from the NumPy experience, all other automatic switching details should
not worry users. But if you want to define a new operator, or implement some
missing operators from NumPy, there are a few steps to follow.

1. Define the gradient of the NumPy version of the operator in
`minpy/array_varians/numpy/numpy_core.py`. There are already a bunch of operators
with patterns to follow. In essence you need to define a second order function
that takes all inputs to the original function, and the gradient passed down as well.
Your job is to have the function pass the gradient through the operator.
1. Follow the same steps for the MXNet version in `minpy/array_variants/mxnet/mxnet_core.py`.
Sometimes the MXNet operator has a slightly different name, then you should `register`
it under function `register_primitives`.

You could technically define only NumPy version or MXNet version. The policy is
smart enought to fall back. But there will be performance penalty copying data
back and forth.

### Using MXNet's symbolic layers

### Using Caffe-defined layers
