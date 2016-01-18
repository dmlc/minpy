#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper for NumPy namespace."""
import types
import functools
import numpy as np

from .. import core
from .. import registry


def unbox_args(f):
    return functools.wraps(f)(lambda *args, **kwargs: f(*args, **kwargs))


def wrap_namespace(ns, t, registry):
    """Wrap namespace into a registry.

    Args:
        ns: Namespace from which functions are to be wrapped.
        t: Type of function.
        registry: Registry into which functions are to be wrapped.
    """
    unchanged_types = {float, int, type(None), type}
    int_types = {np.int, np.int8, np.int16, np.int32, np.int64, np.integer}
    function_types = {np.ufunc, types.FunctionType, types.BuiltinFunctionType}

    for name, obj in ns.items():
        if type(obj) in function_types:
            registry.registry(name, core.Primitive(obj), t)
        elif type(obj) is type and obj in int_types:
            registry.registry(name, obj, t)
        elif type(obj) in unchanged_types:
            registry.registry(name, obj, t)

wrap_namespace(np.__dict__, registry.function_registry,
               registry.FunctionType.NUMPY)
