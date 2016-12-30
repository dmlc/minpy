#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member, invalid-name, undefined-variable
"""Definition of grads of mxnet core functions"""
from __future__ import absolute_import

import operator
import mxnet as mx
from mxnet import _ndarray_internal as _in
from mxnet.ndarray import NDArray
from . import mxnet_wrapper


def _unbroadcast(ans, x, gradfun):
    """Append the gradient function with an unbroadcast function."""
    #pylint: disable= missing-docstring
    if isinstance(ans, NDArray) and isinstance(x, NDArray):
        padded_shape = (1, ) * (len(ans.shape) - len(x.shape)) + x.shape

        def newgradfun(g):
            gg = gradfun(g)
            for axis, (i, j) in enumerate(zip(g.shape, padded_shape)):
                if i != j:
                    gg = mx.nd.sum(gg, axis=axis, keepdims=True)
            if gg.shape != x.shape:
                gg = gg.reshape(x.shape)
            return gg

        return newgradfun
    elif isinstance(ans, NDArray):  # x is numerical value

        def newgradfun(g):
            gg = gradfun(g)
            return mx.nd.sum(gg)
    else:  # both ans and x are numerical value
        return gradfun
    #pylint: enable= missing-docstring

def _maximum_grad_gen0(ans, x, _):
    """Generate gradient function of maximum on lhs."""
    # pylint: disable=protected-access
    return _unbroadcast(ans, x, lambda g: g * _in._equal(x, ans))

def _maximum_grad_gen1(ans, _, y):
    """Generate gradient function of maximum on rhs."""
    # pylint: disable=protected-access
    return _unbroadcast(ans, y, lambda g: g * _in._equal(y, ans))

def _sum_grad(ans, x, axis=None, keepdims=False):
    """ Generate gradient function of sum """
    # pylint: disable=unused-argument
    if axis is None:
        def grad(g):
            """Gradient for scalar."""
            if isinstance(g, (float, int)):
                return mx.nd.full(x.shape, g, x.context)
            else:
                return mx.nd.full(x.shape, g.asscalar(), x.context)

        return grad
    if isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)
    ans_shape_expanded = list(x.shape)
    for a in axis:
        ans_shape_expanded[a] = 1
    xshape = x.shape  # Only shape is needed, hope array `x` could be GC'ed.
    return lambda g: mx.nd.zeros(xshape, ctx=g.context) + g.reshape(tuple(ans_shape_expanded))
    # pylint: enable=unused-argument

def _softmax_output(x, _1):
    """Softmax output implementation."""
    probs = mx.nd.exp(x - mx.nd.max(x, axis=1, keepdims=True))
    probs /= mx.nd.sum(probs, axis=1, keepdims=True)
    return probs

def _softmax_output_grad(ans, x, y):
    """Gradient function for softmax output."""
    def grad(_0): #pylint: disable= missing-docstring
        N = x.shape[0]
        return (ans - y) / N
    return grad

################################################################
# Functions exposed for primitive & gradient registry
def register_primitives(reg, prim_wrapper):
    """ Register primitives in mxnet package """
    mxnet_wrapper.wrap_namespace(mx.nd.__dict__, reg, prim_wrapper)
    # Additional primitives due to naming issues in MXNet.
    reg.register('reshape', prim_wrapper(NDArray.reshape))
    reg.register('softmax_output', prim_wrapper(_softmax_output))


def def_grads(prims):
    """ Define gradient function for primitives """
    identity = lambda x: x
    # dot
    prims('dot').def_grad(lambda ans, a, b: lambda g: mx.nd.dot(g, b.T))
    prims('dot').def_grad(
        lambda ans, a, b: lambda g: mx.nd.dot(a.T, g), argnum=1)
    # non-linear
    prims('tanh').def_grad(lambda ans, x: lambda g: g * (1 - ans ** 2))
    prims('exp').def_grad(lambda ans, x: lambda g: g * ans)
    prims('log').def_grad(lambda ans, x: lambda g: g / x)
    # reduce
    prims('sum').def_grad(_sum_grad)
    # + - * /
    prims('multiply').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * y))
    prims('multiply').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: x * g), argnum=1)
    prims('add').def_grad(lambda ans, x, y: _unbroadcast(ans, x, identity))
    prims('add').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, identity), argnum=1)
    prims('subtract').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, identity))
    prims('subtract').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, operator.neg), argnum=1)
    prims('divide').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g / y))
    prims('divide').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: -g * x / (y * y)),
        argnum=1)
    prims('true_divide').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g / y))
    prims('true_divide').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: -g * x / (y * y)),
        argnum=1)
    prims('maximum').def_grad(_maximum_grad_gen0)
    prims('maximum').def_grad(_maximum_grad_gen1, argnum=1)
    # negate
    prims('negative').def_grad(lambda ans, x: operator.neg)
    prims('transpose').def_grad(lambda ans, x: mx.nd.transpose)
    prims('abs').def_grad(lambda ans, x: lambda g: mx.nd.sign(x) * g)
    prims('sign').def_grad_zero()
    prims('round').def_grad_zero()
    prims('ceil').def_grad_zero()
    prims('floor').def_grad_zero()
    prims('sqrt').def_grad(lambda ans, x: lambda g: g * 0.5 / mx.nd.sqrt(x))
    prims('sin').def_grad(lambda ans, x: lambda g: g * mx.nd.cos(x))
    prims('cos').def_grad(lambda ans, x: lambda g: -g * mx.nd.sin(x))
    prims('power').def_grad(
        lambda ans, x, y: _unbroadcast(ans, x, lambda g: g * y * mx.nd.power(x, y - 1))
    )
    prims('power').def_grad(
        lambda ans, x, y: _unbroadcast(ans, y, lambda g: g * mx.nd.log(x) * ans),
        argnum=1)
    prims('reshape').def_grad(
        lambda _0, x, _1: lambda g: NDArray.reshape(g, x.shape))
    prims('expand_dims').def_grad(
        lambda ans, x, axis: lambda g: NDArray.reshape(g, x.shape))
    prims('softmax_output').def_grad(_softmax_output_grad)
