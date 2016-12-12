""" Initializer codes """
import numpy
import minpy.numpy as np
import minpy.numpy.random as npr

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

    fan_out = shape[0]
    if len(shape) > 1:
        fan_in = numpy.prod(shape[1:])
    else:
        fan_in = 0
    var = numpy.sqrt(6.0 / (fan_out + fan_in))
    ret = npr.randn(*shape) * var
    return ret


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
    config.setdefault('value', 0.0)
    val = config['value']
    return np.ones(shape) * val


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
    config.setdefault('mu', 0.0)
    config.setdefault('stdvar', 0.001)
    stdvar = config['stdvar']
    meanvar = config['mu']
    return npr.randn(*shape) * stdvar + meanvar


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
    func = config.setdefault('function', np.zeros)
    ret = func(shape)
    return ret
