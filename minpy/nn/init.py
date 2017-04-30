""" Initializer codes """
from functools import reduce as _reduce
from operator import mul as _mul

import mxnet.ndarray as _nd

# pylint: disable=no-member


def xavier(shape, _):
    """Initialize weights with xavier initializer.

    Xavier initializer init matrix based on fan_in and fan_out

    Parameters
    ----------
    shape : tuple
        Shape of the array to be initialized.
    _ : placeholder

    Returns
    -------
    Array
        Initialized array of size `shape`.

    """

    fan_in = _reduce(_mul, shape[1:]) if len(shape) > 1 else 0
    fan_out = shape[0]

    scale = (6.0 / (fan_in + fan_out)) ** 0.5
    return _nd.random_normal(scale=var, shape=shape)


def constant(shape, config):
    """Initialize weights with constant value.

    Parameters
    ----------
    shape : tuple
        Shape of the array to be initialized.
    config : dict
        The value to initailize the array

    Returns
    -------
    Array
        Initialized array of size `shape` and with the value `value`

    """

    return _nd.ones(shape) * config.setdefault('value', 0.0)


def gaussian(shape, config):
    """Initialize weights with gaussian distribution.

    Parameters
    ----------
    shape : tuple
        Shape of the array to be initialized.
    config : dict
        Mean and standard variance of the distribution

    Returns
    -------
    Array
        Initialized array of size `shape`

    """
    mu = config.setdefault('mu', 0.0)
    stdvar = config.setdefault('stdvar', 0.001)
    return _nd.random_normal(loc=mu, scale=stdvar, shape=shape)


def custom(shape, config):
    """Initialize weights with a user-defined function.

    The function is provided via `config['function']`, and should be a function that receives
    a shape tuple and returns an initialized `Array` with that shape.

    Parameters
    ----------
    shape : tuple
        Shape of the array to be initialized.
    config : dict
        Configuration parameters. Set a user-defined weight initialization function through
        the 'function' key.

    Returns
    -------
    Array
        Initialized array of size `shape`, or an array of zeros if no function was provided.

    """
    func = config.setdefault('function', _nd.zeros)
    ret = func(shape)
    return ret
