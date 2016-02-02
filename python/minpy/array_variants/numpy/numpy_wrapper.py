#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper for NumPy namespace."""
import types
import numpy as np

def wrap_namespace(ns, reg, prim_wrapper):
    """Wrap namespace into a reg.

    :param ns: Namespace from which functions are to be wrapped.
    :param reg: Registry into which functions are to be wrapped.
    :param prim_wrapper: Wrapper to convert a raw function to primitive
    """
    unchanged_types = {float, int, type(None), type}
    int_types = {np.int, np.int8, np.int16, np.int32, np.int64, np.integer}
    function_types = {np.ufunc, types.FunctionType, types.BuiltinFunctionType}

    for name, obj in ns.items():
        if type(obj) in function_types:
            prim = prim_wrapper(obj)
            reg.register(name, prim)
