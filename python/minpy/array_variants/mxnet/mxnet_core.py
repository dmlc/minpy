#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Definition of grads of mxnet core functions"""
from __future__ import absolute_import

import operator
import mxnet
from mxnet import ndarray
from . import mxnet_wrapper

def unbroadcast(ans, x, gradfun):
    # TODO currently no broadcasting for mx.ndarray
    return gradfun

def register_primitives(reg, prim_wrapper):
    mxnet_wrapper.wrap_namespace(mxnet.ndarray.__dict__, reg, prim_wrapper)

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
    prims('sum').def_grad(lambda ans, x: lambda g: ndarray.full(x.shape, g, x.context))
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
