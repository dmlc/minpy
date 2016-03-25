#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Definition of grads of mxnet core functions"""
from __future__ import absolute_import

import operator
import mxnet
from mxnet import ndarray
from . import mxnet_wrapper

def unbroadcast(ans, x, gradfun):
    def newgradfun(g):
        gg = g
        for axis, (i, j) in enumerate(zip(g.shape, x.shape)):
            if i != j:
                gg = ndarray.sum(gg, axis=axis, keepdims=True)
        return gradfun(gg)
    return newgradfun

def register_primitives(reg, prim_wrapper):
    mxnet_wrapper.wrap_namespace(mxnet.ndarray.__dict__, reg, prim_wrapper)

def gen_sum_grad(ans, x, axis, keepdims):
    xshape = x.shape
    if axis is None:
        return lambda g: ndarray.full(x.shape, g, x.context)
    if type(axis) is int:
        axis = [axis]
    elif type(axis) is tuple:
        axis = list(axis)
    for a in axis:
        xshape[a] = 1
    def sum_grad(g):
        return ndarray.zeros(x.shape) + g.reshape(xshape)
    return sum_grad

def def_grads(reg, prims):
    def identity(x):
        return x
    # dot
    prims('dot').def_grad(lambda ans, a, b: lambda g: ndarray.dot(g, b.T))
    prims('dot').def_grad(lambda ans, a, b: lambda g: ndarray.dot(a.T, g), argnum=1)
    # non-linear
    #prims.tanh.def_grad(lambda ans, x: lambda g: g / np.cosh(x) ** 2)
    prims('exp').def_grad(lambda ans, x: lambda g: g * ans)
    prims('log').def_grad(lambda ans, x: lambda g: g / x)
    # reduce
    prims('sum').def_grad(lambda ans, x, axis=None, keepdims=False: gen_sum_grad(ans, x, axis, keepdims))
    # + - * /
    prims('multiply').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y))
    prims('multiply').def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: x * g), argnum=1)
    prims('add').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('add').def_grad(lambda ans, x, y: unbroadcast(ans, y, identity), argnum=1)
    prims('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    prims('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, y, operator.neg), argnum=1)
    prims('divide').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
    prims('divide').def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: - g * x / (y * y)), argnum=1)
    prims('true_divide').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
    prims('true_divide').def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: - g * x / (y * y)), argnum=1)
    # power
    #prims.power.def_grad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * y * x ** (y - 1)))
    #prims.power.def_grad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * ndarray.log(x) * x ** y), argnum=1)
    # mod
    #prims.mod.def_grad(lambda ans, x, y : unbroadcast(ans, x, identity))
    #prims.mod.def_grad(lambda ans, x, y : unbroadcast(ans, y, lambda g : - g * ndarray.floor(x/y)), argnum=1)
    # negate
    prims('negative').def_grad(lambda ans, x: operator.neg)
