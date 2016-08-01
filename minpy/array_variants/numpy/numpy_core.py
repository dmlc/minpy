#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= invalid-name
"""Definition of grads of core functions"""
from . import numpy_wrapper

import operator
import numpy as np


def _minpy_getitem(arr, index):
    """ Slice operation """
    return arr[index]


def _minpy_getitem_grad(arr, index, g):
    """ Gradient of slice operation """
    ret = np.zeros_like(arr)
    np.add.at(ret, index, g)
    return ret


def _unbroadcast(ans, x, gradfun):
    """Unbroadcast to original shape.

    :param ans: Output of forward function (broadcasted shape).
    :param x: Input of forward function (pre-broadcasted shape).
    :param gradfun: Gradient function.
    :return: New gradient function with proper unbroadcast on the result.
    """
    #pylint: disable= missing-docstring
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
    elif isinstance(ans, np.ndarray):  # x is numerical value
        new_fun = lambda g: np.sum(gradfun(g))
    else:  # both ans and x are numerical value
        return gradfun
    new_fun.__name__ = 'unbroadcast_{0}'.format(gradfun.__name__)
    return new_fun
    #pylint: enable= missing-docstring


def _sum_grad(ans, x, axis=None, keepdims=False):
    """ Generate gradient function of sum """
    if axis is None:
        return lambda g: np.full(x.shape, g)
    if isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)
    ans_shape_expanded = list(x.shape)
    for a in axis:
        ans_shape_expanded[a] = 1
    xshape = x.shape  # Only shape is needed, hope array `x` could be GC'ed.
    return lambda g: np.zeros(xshape) + np.reshape(g, ans_shape_expanded)

import sys
def _chooser_grad(ans, x, axis=None, keepdims=False):
    """ Generate gradient function of max """
    #pylint: disable= missing-docstring
    print('x', x)
    print('ans', ans)
    if axis is None:
        # Reduce for all axis.
        axis = list(range(len(x.shape)))
    elif isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)
    ans_shape_expanded = list(x.shape)
    for a in axis:
        ans_shape_expanded[a] = 1
    # Find locations of the answer elements.
    ans_repeated = np.zeros(x.shape) + np.reshape(ans, ans_shape_expanded)
    locations = ans_repeated == x
    print('locations', locations)
    xshape = x.shape  # Only shape is needed, hope array `x` could be GC'ed.
    def _gradfun(g):
        g_repeated = np.zeros(xshape) + np.reshape(g, ans_shape_expanded)
        ret = g_repeated * locations
        print('g', g)
        print('ret', ret)
        sys.exit(0)
    return _gradfun
    #pylint: enable= missing-docstring

################################################################
# Functions exposed for primitive & gradient registry
def register_primitives(reg, prim_wrapper):
    """ Register primitives in numpy """
    numpy_wrapper.wrap_namespace(np.__dict__, reg, prim_wrapper)
    # additional primitives
    reg.register('_minpy_getitem', prim_wrapper(_minpy_getitem))

def def_grads(reg, prims):
    """ Define gradient function for primitives """
    identity = lambda x: x
    # Dot.
    prims('dot').def_grad(lambda ans, a, b: lambda g: np.dot(g, b.T))
    prims('dot').def_grad(lambda ans, a, b: lambda g: np.dot(a.T, g), argnum=1)

    # Unary functions.
    prims('tanh').def_grad(lambda ans, x: lambda g: g / np.cosh(x)**2)
    prims('log').def_grad(lambda ans, x: lambda g: g / x)
    prims('exp').def_grad(lambda ans, x: lambda g: ans * g)

    # Reducing functions.
    prims('sum').def_grad(_sum_grad)
    #prims('max').def_grad(_chooser_grad)
    #prims('min').def_grad(_chooser_grad)
    #prims('amax').def_grad(_chooser_grad)
    #prims('amin').def_grad(_chooser_grad)
    prims('max').def_grad_zero()
    prims('min').def_grad_zero()
    prims('amax').def_grad_zero()
    prims('amin').def_grad_zero()

    # Binary functions
    prims('multiply').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * y))
    prims('multiply').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: x * g),
        argnum=1)
    prims('add').def_grad(lambda ans, x, y: _unbroadcast(ans, x, identity))
    prims('add').def_grad(lambda ans, x, y: _unbroadcast(ans, y, identity),
                          argnum=1)
    prims('subtract').def_grad(lambda ans, x, y: _unbroadcast(ans, x, identity))
    prims('subtract').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, operator.neg),
        argnum=1)
    prims('divide').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g / y))
    prims('divide').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: -g * x / y**2),
        argnum=1)
    prims('true_divide').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g / y))
    prims('true_divide').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: -g * x / y**2),
        argnum=1)
    prims('power').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * y * x**(y - 1)))
    prims('power').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: g * np.log(x) * x**y),
        argnum=1)
    prims('mod').def_grad(lambda ans, x, y: _unbroadcast(ans, x, identity))
    prims('mod').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: -g * np.floor(x / y)),
        argnum=1)
    prims('negative').def_grad(lambda ans, x: operator.neg)
    prims('transpose').def_grad(lambda ans, x: np.transpose)
    prims('abs').def_grad(lambda ans, x: lambda g: np.sign(x) * g)
    prims('sign').def_grad_zero()
    prims('round').def_grad_zero()
    prims('ceil').def_grad_zero()
    prims('floor').def_grad_zero()
    prims('sqrt').def_grad(lambda ans, x: lambda g: g * 0.5 / np.sqrt(x))
    prims('sin').def_grad(lambda ans, x: lambda g: g * np.cos(x))
    prims('cos').def_grad(lambda ans, x: lambda g: -g * np.sin(x))
    prims('power').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * y * np.power(x, y - 1)))
    prims('power').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: g * np.log(x) * ans),
        argnum=1)
    prims('maximum').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * (x == ans)))
    prims('maximum').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: g * (y == ans)),
        argnum=1)
    prims('minimum').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * (x == ans)))
    prims('minimum').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: g * (y == ans)),
        argnum=1)
    prims('_minpy_getitem').def_grad(
        lambda ans, x, index: lambda g: _minpy_getitem_grad(x, index, g))
    prims('reshape').def_grad(
        lambda ans, x, _1: lambda g: np.reshape(g, x.shape))
    prims('append').def_grad(
        lambda ans, arr, values, axis=None: lambda g: np.split(g, [arr.shape[axis]], axis)[0])
    prims('append').def_grad(
        lambda ans, arr, values, axis=None: lambda g: np.split(g, [arr.shape[axis]], axis)[1],
        argnum=1)
