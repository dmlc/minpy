""" Initializer codes """
import minpy
import minpy.numpy as np
import minpy.numpy.random as npr
import numpy

def xavier(shape, config=None):
    if len(shape) > 1:
        fan_in = numpy.prod(shape[1:])
    else:
        fan_in = 0
    var = 6.0 / (shape[0] + fan_in)
    ret = npr.randn(*shape) * var
    return ret


def constant(shape, config=None):
    val = 0
    if config is not None:
        val = config['value']
    return np.ones(shape) * val
