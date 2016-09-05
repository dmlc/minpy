""" Initializer codes """
import minpy.numpy as np
import minpy.numpy.random as npr
import numpy

def xavier(shape, config):
    fan_out = shape[0]
    if len(shape) > 1:
        fan_in = numpy.prod(shape[1:])
    else:
        fan_in = 0
    var = numpy.sqrt(6.0 / (fan_out + fan_in))
    ret = npr.randn(*shape) * var
    return ret


def constant(shape, config):
    config.setdefault('value', 0.0)
    val = config['value']
    return np.ones(shape) * val


def gaussian(shape, config):
    config.setdefault('mu', 0.0)
    config.setdefault('stdvar', 0.001)
    stdvar = config['stdvar']
    mu = config['mu']
    return npr.randn(*shape) * stdvar + mu


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
    f = config.setdefault('function', np.zeros)
    ret = f(shape)
    return ret
