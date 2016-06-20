""" Initializer codes """
import minpy
import minpy.numpy as np
import minpy.numpy.random as npr

def xavier(shape, config=None):
    var = len(shape) / sum(shape)
    return npr.randn(*shape) * var

def constant(shape, config=None):
    val = 0
    if config is not None:
        val = config['value']
    return np.ones(shape) * val
