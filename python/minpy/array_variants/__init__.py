#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable= invalid-name
""" Package for different implementations of array computations """
from __future__ import absolute_import

from ..utils import common
from . import numpy
from . import mxnet

class ArrayType:
    """Enumeration of types of arrays."""
    NUMPY = 0
    MXNET = 1

variants = {'numpy': ArrayType.NUMPY, 'mxnet': ArrayType.MXNET}
array_types = {'numpy': numpy.array_type, 'mxnet': mxnet.array_type}
number_types = {'native': [int, float], 'numpy': numpy.number_type, 'mxnet': mxnet.number_type}
