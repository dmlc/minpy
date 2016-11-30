#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper for NumPy namespace."""
from __future__ import absolute_import
from __future__ import print_function

import types
import numpy as np


def wrap_namespace(nspace, reg, prim_wrapper):
    """Register all functions in a given namespace in the primitive registry.

    :param nspace: Namespace from which functions are to be registered.
    :param reg: Primitive registry.
    :param prim_wrapper: Wrapper to convert a raw function to primitive.
    """
    function_types = {np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in nspace.items():
        if type(obj) in function_types:  # pylint: disable=unidiomatic-typecheck
            prim = prim_wrapper(obj)
            reg.register(name, prim)
