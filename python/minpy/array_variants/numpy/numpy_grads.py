#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gradient definitions for NumPy."""
import operator
import numpy as np

from . import numpy_wrapper as npw
from . import random


def unbroadcast(ans, x, gradfun):
    """Unbroadcast to original shape.

    Args:
        ans: Data to unbroadcast.
        x: Original data.
        gradfun: Gradient function.

    Returns:
        Result with original shape.
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


def identity(x):
    return x

# Dot.
npw.dot.def_grad(lambda ans, a, b: lambda g: np.dot(g, b.T))
npw.dot.def_grad(lambda ans, a, b: lambda g: np.dot(a.T, g), argnum=1)

# Nonlinear functions.
npw.tanh.def_grad(lambda ans, x: lambda g: g / np.cosh(x) ** 2)
npw.log.def_grad(lambda ans, x: lambda g: g / x)

npw.sum.def_grad(lambda ans, x: lambda g: np.full(x.shape, g))
npw.multiply.def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y))
npw.multiply.def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: x * g),
                      argnum=1)
npw.add.def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
npw.add.def_grad(lambda ans, x, y: unbroadcast(ans, y, identity), argnum=1)
npw.subtract.def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
npw.subtract.def_grad(lambda ans, x, y: unbroadcast(ans, y, operator.neg),
                      argnum=1)
npw.divide.def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
npw.divide.def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                                  lambda g: -g * x / y ** 2),
                    argnum=1)
npw.true_divide.def_grad(lambda ans, x, y: unbroadcast(ans, x,
                                                       lambda g: g / y))
npw.true_divide.def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                                       lambda g: -g * x /
                                                       y ** 2),
                         argnum=1)
npw.power.def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y *
                                                 x ** (y - 1)))
npw.power.def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: g *
                                                 np.log(x) * x ** y), argnum=1)
npw.mod.def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
npw.mod.def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                               lambda g: -g * np.floor(x / y)),
                 argnum=1)
npw.negative.def_grad(lambda ans, x: operator.neg)
random.random.def_grad_zero()
random.randn.def_grad_zero()
