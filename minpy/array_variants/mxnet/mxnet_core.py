#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= no-member, invalid-name
"""Definition of grads of mxnet core functions"""
from __future__ import absolute_import

import operator
import mxnet
from mxnet import ndarray
from mxnet.ndarray import NDArray
from . import mxnet_wrapper


def unbroadcast(ans, x, gradfun):
    """ Append the gradient function with an unbroadcast function """
    #pylint: disable= missing-docstring
    if isinstance(ans, NDArray) and isinstance(x, NDArray):
        padded_shape = (1,) * (len(ans.shape) - len(x.shape)) + x.shape
        def newgradfun(g):
            gg = gradfun(g)
            for axis, (i, j) in enumerate(zip(g.shape, padded_shape)):
                if i != j:
                    gg = ndarray.sum(gg, axis=axis, keepdims=True)
            if gg.shape != x.shape:
                gg = gg.reshape(x.shape)
            return gg
        return newgradfun
    elif isinstance(ans, NDArray):  # x is numerical value
        def newgradfun(g):
            gg = gradfun(g)
            return ndarray.sum(gg)
    else:  # both ans and x are numerical value
        return gradfun
    #pylint: enable= missing-docstring


def register_primitives(reg, prim_wrapper):
    """ Register primitives in mxnet package """
    mxnet_wrapper.wrap_namespace(mxnet.ndarray.__dict__, reg, prim_wrapper)
    # Additional primitives due to naming issues in MXNet.
    reg.register('reshape', prim_wrapper(NDArray.reshape))


def gen_sum_grad(ans, x, axis, keepdims):
    """ Generate gradient function of sum """
    xshape = list(x.shape)
    if axis is None:
        def grad(g):
            if isinstance(g, float) or isinstance(g, int):
                return ndarray.full(x.shape, g, x.context)
            else:
                return ndarray.full(x.shape, g.asscalar(), x.context)
        return grad
    if isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)
    for a in axis:
        xshape[a] = 1
    return lambda g: ndarray.zeros(x.shape, ctx=g.context) + g.reshape(tuple(xshape))


def def_grads(reg, prims):
    """ Define gradient function for primitives """
    identity = lambda x: x
    # dot
    prims('dot').def_grad(lambda ans, a, b: lambda g: ndarray.dot(g, b.T))
    prims('dot').def_grad(lambda ans, a, b: lambda g: ndarray.dot(a.T, g),
                          argnum=1)
    # non-linear
    #prims.tanh.def_grad(lambda ans, x: lambda g: g / np.cosh(x) ** 2)
    prims('exp').def_grad(lambda ans, x: lambda g: g * ans)
    prims('log').def_grad(lambda ans, x: lambda g: g / x)
    # reduce
    prims('sum').def_grad(
        lambda ans, x, axis=None, keepdims=False: gen_sum_grad(
            ans,
            x,
            axis,
            keepdims))
    # + - * /
    prims('multiply').def_grad(
        lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y))
    prims('multiply').def_grad(
        lambda ans, x, y: unbroadcast(ans, y, lambda g: x * g),
        argnum=1)
    prims('add').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('add').def_grad(lambda ans, x, y: unbroadcast(ans, y, identity),
                          argnum=1)
    prims('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('subtract').def_grad(
        lambda ans, x, y: unbroadcast(ans, y, operator.neg),
        argnum=1)
    prims('divide').def_grad(
        lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
    prims('divide').def_grad(
        lambda ans, x, y: unbroadcast(ans, y, lambda g: -g * x / (y * y)),
        argnum=1)
    prims('true_divide').def_grad(
        lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
    prims('true_divide').def_grad(
        lambda ans, x, y: unbroadcast(ans, y, lambda g: -g * x / (y * y)),
        argnum=1)
    # mod
    #prims.mod.def_grad(lambda ans, x, y : unbroadcast(ans, x, identity))
    #prims.mod.def_grad(
    #lambda ans, x, y : unbroadcast(ans, y, lambda g : - g * ndarray.floor(x/y)), argnum=1)
    # negate
    prims('negative').def_grad(lambda ans, x: operator.neg)
    prims('transpose').def_grad(lambda ans, x: mxnet.nd.transpose)
    prims('abs').def_grad(lambda ans, x: lambda g: mxnet.nd.sign(x) * g)
    prims('sign').def_grad_zero()
    prims('round').def_grad_zero()
    prims('ceil').def_grad_zero()
    prims('floor').def_grad_zero()
    prims('sqrt').def_grad(lambda ans, x: lambda g: g * 0.5 / mxnet.nd.sqrt(x))
    prims('sin').def_grad(lambda ans, x: lambda g: g * mxnet.nd.cos(x))
    prims('cos').def_grad(lambda ans, x: lambda g: -g * mxnet.nd.sin(x))
    prims('power').def_grad(
        lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y * mxnet.nd.power(x, y - 1)))
    prims('power').def_grad(
        lambda ans, x, y: unbroadcast(ans, y, lambda g: g * mxnet.nd.log(x) * ans),
        argnum=1)
    prims('reshape').def_grad(
        lambda _0, x, _1: lambda g: mxnet.nd.NDArray.reshape(g, x.shape))
