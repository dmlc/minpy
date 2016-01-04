#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gradient definitions for NumPy."""
import operator
import numpy as np

from . import numpy_wrapper as npw
from . import random
from .. import array


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


class NumpyNode(array.Array):

    def __init__(self, val):
        super().__init__(val)

    @property
    def shape(self):
        return self._val.shape

    def __neg__(self):
        return npw.negative(self)

    def __add__(self, other):
        return npw.add(self, other)

    def __sub__(self, other):
        return npw.subtract(self, other)

    def __mul__(self, other):
        return npw.multiply(self, other)

    def __div__(self, other):
        return npw.divide(self, other)

    def __truediv__(self, other):
        return npw.true_divide(self, other)

    def __pow__(self, other):
        return npw.power(self, other)

    def __mod__(self, other):
        return npw.mod(self, other)

    def __radd__(self, other):
        return npw.add(other, self)

    def __rsub__(self, other):
        return npw.subtract(other, self)

    def __rmul__(self, other):
        return npw.multiply(other, self)

    def __rdiv__(self, other):
        return npw.divide(other, self)

    def __rtruediv__(self, other):
        return npw.true_divide(other, self)

    def __rpow__(self, other):
        return npw.power(other, self)

    def __rmod__(self, other):
        return npw.mod(other, self)
