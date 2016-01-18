#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper for NumPy random functions."""
import minpy.registry as registry
import minpy.array_variants as variants
from . import numpy_wrapper

from numpy import random

print 'power' in random.__dict__

numpy_wrapper.wrap_namespace(random.__dict__, registry.function_registry,
               variants.FunctionType.NUMPY)

def def_grads(reg):
    def get(name):
        return reg.get(name, variants.FunctionType.NUMPY)
    get('random').def_grad_zero()
    get('randn').def_grad_zero()

def_grads(registry.function_registry)
