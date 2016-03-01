#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Definition of grads of core functions"""
from . import numpy_wrapper

import operator
import numpy as np

def unbroadcast(ans, x, gradfun):
    """Unbroadcast to original shape.

    :param ans: Data to broadcast.
    :param x: Original data.
    :param gradfun: Gradient function.
    :return: Result with original shape.
    """
    if isinstance(x, np.ndarray):
        shape = x.shape

        def new_fun(g):
            result = gradfun(g)
            while len(shape) < np.ndim(result):
                result = np.sum(result, axis=0)
            for axis, size in enumerate(shape):
                if size == 1:
                    result = np.sum(result, axis=axis, keepdims=True)
            assert np.shape(result) == shape
            return result
    elif isinstance(ans, np.ndarray):
        new_fun = lambda g: np.sum(gradfun(g))
    else:
        return gradfun
    new_fun.__name__ = 'unbroadcast_{0}'.format(gradfun.__name__)
    return new_fun

def register_primitives(reg, make_prim):
    numpy_wrapper.wrap_namespace(np.__dict__, reg, make_prim)

def def_grads(reg, prims):
    def identity(x):
        return x
    # Dot.
    prims('dot').def_grad(lambda ans, a, b: lambda g: np.dot(g, b.T))
    prims('dot').def_grad(lambda ans, a, b: lambda g: np.dot(a.T, g), argnum=1)

    # Nonlinear functions.
    prims('tanh').def_grad(lambda ans, x: lambda g: g / np.cosh(x) ** 2)
    prims('log').def_grad(lambda ans, x: lambda g: g / x)
    prims('exp').def_grad(lambda ans, x: lambda g: ans * g)

    prims('sum').def_grad(lambda ans, x: lambda g: np.full(x.shape, g))
    prims('multiply').def_grad(lambda ans, x,
                             y: unbroadcast(ans, x, lambda g: g * y))
    prims('multiply').def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: x * g),
                             argnum=1)
    prims('add').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('add').def_grad(lambda ans, x, y: unbroadcast(ans, y, identity), argnum=1)
    prims('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, y, operator.neg),
                             argnum=1)
    prims('divide').def_grad(lambda ans, x,
                           y: unbroadcast(ans, x, lambda g: g / y))
    prims('divide').def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                                         lambda g: -g * x / y ** 2),
                           argnum=1)
    prims('true_divide').def_grad(lambda ans, x, y: unbroadcast(ans, x,
                                                              lambda g: g / y))
    prims('true_divide').def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                                              lambda g: -g * x /
                                                              y ** 2),
                                argnum=1)
    prims('power').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y *
                                                        x ** (y - 1)))
    prims('power').def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: g *
                                                        np.log(x) * x ** y), argnum=1)
    prims('mod').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('mod').def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                                      lambda g: -g * np.floor(x / y)),
                        argnum=1)
    prims('negative').def_grad(lambda ans, x: operator.neg)

#def_grads(registry.function_registry)
