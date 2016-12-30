#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, undefined-variable
"""Definition of grads of core functions for numpy implementation"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import operator

import numpy as np

from minpy.array_variants.numpy import numpy_wrapper

def _identity(x):
    """ identity function lambda x: x """
    return x


def _minpy_getitem(arr, index):
    """ Slice operation """
    return arr[index]


def _minpy_getitem_grad(arr, index, g):
    """ Gradient of slice operation """
    ret = np.zeros_like(arr)
    np.add.at(ret, index, g)
    return ret


# TODO: Collect customized functions into a separate module.
def _sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


# TODO: Refine this function after MXNet's refinement.
def _onehot_encode(indices, out):
    """One hot encoding indices into matrix out. NumPy equivalence.

    Parameters
    ----------
    indices: ndarray
        An all zero ndarray containing indices of the categorical features.
    out: ndarray
        The result holder of the encoding.
    Returns
    -------
    out: Array
        Same as out.
    """
    N = indices.shape[0]
    out[np.arange(N), indices] = 1
    return out


def _chooser_grad(ans, a, axis=None, _0=None, keepdims=False):
    """ Gradient of amax function """
    repeater, _ = _match_shape(a, axis, keepdims=keepdims)
    argmax_locations = a == repeater(ans)
    return lambda g: repeater(g) * argmax_locations


def _match_shape(a, axis, keepdims):
    """ Return a function repeats input to match a given shape

    This function generates a function f l: m, n that repeats
    l along axis to match the shape of x. l must be compatible
    with the arguments.

    :param a: the array template for f to match
    :param axis: the axis for f to repeat in order to match x
    :type axis: None, or Int, or Tuple
    :param keepdims: True if the same dim of x, l is expected
    :type keepdims: Bool
    :return: function f, number of repetition
    """
    assert isinstance(axis, (type(None), int,
                             tuple)), "axis must be None, int, or tuple."

    if not isinstance(a, np.ndarray):
        return _identity, 1
    shape = a.shape
    if axis is None:
        # np.full() has a bug for complex numbers, explicit type is needed
        if np.iscomplexobj(a):
            dtype = a.dtype
        else:
            dtype = None
        return lambda g: np.full(shape, g, dtype=dtype), np.prod(shape)
    elif isinstance(axis, int):
        if keepdims:
            return lambda g: np.repeat(g, shape[axis], axis), shape[axis]
        else:
            return lambda g: np.repeat(np.expand_dims(g, axis),
                                       shape[axis], axis), shape[axis]
    else:
        repeats = [shape[i] if i in axis else 1 for i in xrange(len(shape))]
        expanded = [
            shape[i] if i not in axis else 1 for i in xrange(len(shape))
        ]
        num_reps = np.prod(np.array(shape)[list(axis)])

        if keepdims:
            return lambda g: np.tile(g, repeats), num_reps
        else:
            return lambda g: np.tile(np.reshape(g, expanded), repeats), num_reps


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


def _sum_grad(_0, x, axis=None, keepdims=False):  # pylint: disable=unused-argument
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

def _softmax_output(x, _1):
    """Softmax output implementation."""
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

def _softmax_output_grad(ans, x, y):
    """Gradient function for softmax output."""
    def grad(_0): #pylint: disable= missing-docstring
        N = x.shape[0]
        return (ans - y) / N
    return grad

# TODO: Clean one of the implementations
# import sys
# def _chooser_grad(ans, x, axis=None, keepdims=False):
#     """ Generate gradient function of max """
#     #pylint: disable= missing-docstring
#     print('x', x)
#     print('ans', ans)
#     if axis is None:
#         # Reduce for all axis.
#         axis = list(range(len(x.shape)))
#     elif isinstance(axis, int):
#         axis = [axis]
#     elif isinstance(axis, tuple):
#         axis = list(axis)
#     ans_shape_expanded = list(x.shape)
#     for a in axis:
#         ans_shape_expanded[a] = 1
#     # Find locations of the answer elements.
#     ans_repeated = np.zeros(x.shape) + np.reshape(ans, ans_shape_expanded)
#     locations = ans_repeated == x
#     print('locations', locations)
#     xshape = x.shape  # Only shape is needed, hope array `x` could be GC'ed.
#     def _gradfun(g):
#         g_repeated = np.zeros(xshape) + np.reshape(g, ans_shape_expanded)
#         ret = g_repeated * locations
#         print('g', g)
#         print('ret', ret)
#         sys.exit(0)
#     return _gradfun
#     #pylint: enable= missing-docstring


################################################################
# Functions exposed for primitive & gradient registry
def register_primitives(reg, prim_wrapper):
    """Register primitives in numpy"""
    numpy_wrapper.wrap_namespace(np.__dict__, reg, prim_wrapper)
    # additional primitives
    reg.register('_minpy_getitem', prim_wrapper(_minpy_getitem))
    reg.register('sigmoid', prim_wrapper(_sigmoid))
    reg.register('onehot_encode', prim_wrapper(_onehot_encode))
    reg.register('softmax_output', prim_wrapper(_softmax_output))


def def_grads(prims):
    """ Define gradient function for primitives """
    # Dot.
    prims('dot').def_grad(lambda ans, a, b: lambda g: np.dot(g, b.T))
    prims('dot').def_grad(lambda ans, a, b: lambda g: np.dot(a.T, g), argnum=1)

    # Unary functions.
    prims('tanh').def_grad(lambda ans, x: lambda g: g / np.cosh(x)**2)
    prims('log').def_grad(lambda ans, x: lambda g: g / x)
    prims('exp').def_grad(lambda ans, x: lambda g: ans * g)

    # Reducing functions.
    prims('sum').def_grad(_sum_grad)
    prims('max').def_grad(_chooser_grad)
    prims('amax').def_grad(_chooser_grad)
    prims('min').def_grad(_chooser_grad)
    prims('amin').def_grad(_chooser_grad)

    # Binary functions
    prims('multiply').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * y))
    prims('multiply').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: x * g), argnum=1)
    prims('add').def_grad(lambda ans, x, y: _unbroadcast(ans, x, _identity))
    prims('add').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, _identity), argnum=1)
    prims('subtract').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, _identity))
    prims('subtract').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, operator.neg), argnum=1)
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
    prims('broadcast_to').def_grad(
        lambda ans, x, shape: _unbroadcast(ans, x, _identity))
    prims('mod').def_grad(lambda ans, x, y: _unbroadcast(ans, x, _identity))
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
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * y * ans / x))
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
    prims('expand_dims').def_grad(
        lambda ans, x, axis: lambda g: np.reshape(g, x.shape))
    prims('sigmoid').def_grad(lambda ans, x: lambda g: g * ans * (1 - ans))
    prims('softmax_output').def_grad(_softmax_output_grad)
