#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core gradient calculation."""
import functools
import operator
from .utils import log
from . import array

_logger = log.get_logger(__name__, log.logging.WARNING)

def grad(func, argnum=0):
    @functools.wraps(func)
    def wrapped(*args):
        def make_array(x):
            return x if isinstance(x, array.Array) else array.Array(x)
        arrays = tuple(map(make_array, args))
        result_array = func(*arrays)
        return arrays[argnum].node.partial_derivative(result_array.node)
    return wrapped
