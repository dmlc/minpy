#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=cell-var-from-loop
"""Wrapper for MXNet namespace."""
import types
# pylint: disable=deprecated-lambda


def wrap_namespace(nspace, reg, prim_wrapper):
    """Register all functions in a given namespace in the primitive registry.

    :param nspace: Namespace from which functions are to be registered.
    :param reg: Primitive registry.
    :param prim_wrapper: Wrapper to convert a raw function to primitive.
    """
    function_types = {types.FunctionType, types.BuiltinFunctionType}
    for name, obj in nspace.items():
        if len(list(filter(lambda x: isinstance(obj, x), function_types))) != 0:
            prim = prim_wrapper(obj)
            reg.register(name, prim)
