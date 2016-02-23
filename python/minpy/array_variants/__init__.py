#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import enum

from ..utils import common
from . import numpy
from . import mxnet

class FunctionType(enum.Enum):
    """Enumeration of types of functions.

    Semantically this is different from :class:`..array.ArrayType`,
    but for now one data type corresponds to one function type.
    """
    NUMPY = 0
    MXNET = 1

class ArrayType(enum.Enum):
    """Enumeration of types of arrays."""
    NUMPY = 0
    MXNET = 1

variants = {
        'numpy': (ArrayType.NUMPY, FunctionType.NUMPY),
        'mxnet': (ArrayType.MXNET, FunctionType.MXNET)
        }

array_types = {
        'numpy': numpy.array_type,
        'mxnet': mxnet.array_type,
}
number_types = {
        'native': [int, float],
        'numpy': numpy.number_type,
        'mxnet': mxnet.number_type,
}
