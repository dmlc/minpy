# pylint: disable=invalid-name, pointless-string-statement
""" Optimizer codes. Adapted from cs231n lab codes. """
import mxnet.ndarray as _nd

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""
# pylint: disable=no-member

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - lr: Scalar learning rate.
    - wd: 
    - rescale_grad:
    - clip_gradient:
    """
    _nd.sgd_update(w, dw, out=w, **config)

    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - mom: A numpy array of the same shape as w and dw used to store a moving
                average of the gradients.
    - lr: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
                Setting momentum = 0 reduces to sgd.
    - wd: 
    - rescale_grad:
    - clip_gradient:
    """
    config.setdefault('mom', _nd.zeros_like(w))

    _nd.sgd_mom_update(w, dw, out=w, **config)

    return w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - n: Moving average of second moments of gradients.
    - lr: Scalar learning rate.
    - gamma1: Scalar between 0 and 1 giving the decay rate for the squared
                  gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - wd: 
    - rescale_grad:
    - clip_gradient:
    - clip_weights:
    """

    if config is None:
        config = {}
    config.setdefault('n', _nd.zeros_like(w))

    _nd.rmsprop_update(w, dw, out=w, **config)

    return w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - mean: Moving average of gradient.
    - var: Moving average of squared gradient.
    - lr: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - wd:
    - rescale_grad:
    - clip_gradient:
    """

    if config is None:
        config = {}
    config.setdefault('mean', _nd.zeros_like(w))
    config.setdefault('var', _nd.zeros_like(w))

    _nd.adam_update(w, dw, out=w, **config)

    return w, config
