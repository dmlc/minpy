#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper for NumPy random functions."""
from __future__ import absolute_import
from __future__ import print_function

from minpy.array_variants.numpy import numpy_wrapper

import numpy

def register_primitives(reg, prim_wrapper):
    """ Register primitives """
    numpy_wrapper.wrap_namespace(numpy.random.__dict__, reg, prim_wrapper)

def def_grads(prims):
    """ Define gradients of primitives """
    prims('random').def_grad_zero()
    prims('randn').def_grad_zero()
