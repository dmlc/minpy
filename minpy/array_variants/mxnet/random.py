#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Definition of grads of mxnet random functions"""
from __future__ import absolute_import

import mxnet
from . import mxnet_wrapper

def register_primitives(reg, prim_wrapper):
    """ Register primitives """
    mxnet_wrapper.wrap_namespace(mxnet.random.__dict__, reg, prim_wrapper)

def def_grads(prims):
    """ Define gradients of primitives """
    prims('normal').def_grad_zero()
    prims('uniform').def_grad_zero()
