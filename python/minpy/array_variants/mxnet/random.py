#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Definition of grads of mxnet random functions"""
from __future__ import absolute_import

from mxnet import random
from . import mxnet_wrapper

def register_primitives(reg, prim_wrapper):
    mxnet_wrapper.wrap_namespace(mxnet.random.__dict__, reg, prim_wrapper)

def def_grads(reg, prims):
    prims('random').def_grad_zero()
    prims('randn').def_grad_zero()
