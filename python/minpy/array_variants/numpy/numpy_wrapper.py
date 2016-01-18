#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper for NumPy namespace."""
import types
import functools
import numpy as np

import minpy.core as core
import minpy.registry as registry
import minpy.array_variants as variants

def unbox_args(f):
    return functools.wraps(f)(lambda *args, **kwargs: f(*args, **kwargs))

def wrap_namespace(ns, reg, t):
    """Wrap namespace into a reg.

    Args:
        ns: Namespace from which functions are to be wrapped.
        reg: Registry into which functions are to be wrapped.
        t: Type of function.
    """
    unchanged_types = {float, int, type(None), type}
    int_types = {np.int, np.int8, np.int16, np.int32, np.int64, np.integer}
    function_types = {np.ufunc, types.FunctionType, types.BuiltinFunctionType}

    for name, obj in ns.items():
        if type(obj) in function_types:
            reg.register(name, core.Primitive(obj), t)
        elif type(obj) is type and obj in int_types:
            reg.register(name, obj, t)
        elif type(obj) in unchanged_types:
            reg.register(name, obj, t)

wrap_namespace(np.__dict__, registry.function_registry,
               variants.FunctionType.NUMPY)

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


def def_grads(reg):
    def identity(x):
        return x
    def get(name):
        return reg.get(name, variants.FunctionType.NUMPY)
    # Dot.
    get('dot').def_grad(lambda ans, a, b: lambda g: np.dot(g, b.T))
    get('dot').def_grad(lambda ans, a, b: lambda g: np.dot(a.T, g), argnum=1)

    # Nonlinear functions.
    get('tanh').def_grad(lambda ans, x: lambda g: g / np.cosh(x) ** 2)
    get('log').def_grad(lambda ans, x: lambda g: g / x)

    get('sum').def_grad(lambda ans, x: lambda g: np.full(x.shape, g))
    get('multiply').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y))
    get('multiply').def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: x * g),
                          argnum=1)
    get('add').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    get('add').def_grad(lambda ans, x, y: unbroadcast(ans, y, identity), argnum=1)
    get('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    get('subtract').def_grad(lambda ans, x, y: unbroadcast(ans, y, operator.neg),
                          argnum=1)
    get('divide').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
    get('divide').def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                                      lambda g: -g * x / y ** 2),
                        argnum=1)
    get('true_divide').def_grad(lambda ans, x, y: unbroadcast(ans, x,
                                                           lambda g: g / y))
    get('true_divide').def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                                           lambda g: -g * x /
                                                           y ** 2),
                             argnum=1)
    get('power').def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y *
                                                     x ** (y - 1)))
    get('power').def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: g *
                                                     np.log(x) * x ** y), argnum=1)
    get('mod').def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
    get('mod').def_grad(lambda ans, x, y: unbroadcast(ans, y,
                                                   lambda g: -g * np.floor(x / y)),
                     argnum=1)
    get('negative').def_grad(lambda ans, x: operator.neg)

def_grads(registry.function_registry)
