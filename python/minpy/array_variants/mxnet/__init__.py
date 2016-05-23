#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable= invalid-name
""" Package for numpy array implementations """
from __future__ import absolute_import

import mxnet
from . import mxnet_core

array_type = mxnet.ndarray.NDArray
number_type = []

register_primitives = mxnet_core.register_primitives
def_grads = mxnet_core.def_grads
