#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=cell-var-from-loop
"""Wrapper for MXNet namespace."""
import inspect


def wrap_namespace(nspace, reg, prim_wrapper):
    """Register all functions in a given namespace in the primitive registry.

    :param nspace: Namespace from which functions are to be registered.
    :param reg: Primitive registry.
    :param prim_wrapper: Wrapper to convert a raw function to primitive.
    """
    for name, obj in nspace.items():
        if inspect.isroutine(obj):
            prim = prim_wrapper(obj)
            reg.register(name, prim)
