#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper for MXNet namespace."""
import types

def wrap_namespace(ns, reg, prim_wrapper):
    """Wrap namespace into a reg.

    :param ns: Namespace from which functions are to be wrapped.
    :param reg: Registry into which functions are to be wrapped.
    :param prim_wrapper: Wrapper to convert a raw function to primitive
    """
    unchanged_types = {float, int, type(None), type}
    function_types = {types.FunctionType, types.BuiltinFunctionType}

    for name, obj in ns.items():
        if type(obj) in function_types:
            prim = prim_wrapper(obj)
            reg.register(name, prim)
