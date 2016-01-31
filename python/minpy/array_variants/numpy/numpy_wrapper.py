#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper for NumPy namespace."""
import types
import numpy as np

from ... import core


def wrap_namespace(ns, reg, t):
    """Wrap namespace into a reg.

    :param ns: Namespace from which functions are to be wrapped.
    :param reg: Registry into which functions are to be wrapped.
    :param t: Type of function.
    """
    unchanged_types = {float, int, type(None), type}
    int_types = {np.int, np.int8, np.int16, np.int32, np.int64, np.integer}
    function_types = {np.ufunc, types.FunctionType, types.BuiltinFunctionType}

    for name, obj in ns.items():
        if type(obj) in function_types:
            prim = core.Primitive(obj)
            prim._type = t
            reg.register(name, prim, t)
        # elif type(obj) is type and obj in int_types:
            #reg.register(name, obj, t)
        # elif type(obj) in unchanged_types:
            #reg.register(name, obj, t)
