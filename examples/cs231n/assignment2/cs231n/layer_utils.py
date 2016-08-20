from cs231n.layers import affine_forward, relu_forward, affine_backward, relu_backward

#from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
    a, af_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (af_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
  Backward pass for the affine-relu convenience layer
  """
    af_cache, relu_cache = cache
    daf = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(daf, af_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache
