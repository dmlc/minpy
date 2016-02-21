#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import enum

from ..utils import common
from . import numpy
from . import mxnet

@enum.unique
class FunctionType(enum.Enum):
    """Enumeration of types of functions.

    Semantically this is different from :class:`..array.ArrayType`,
    but for now one data type corresponds to one function type.
    """
    NUMPY = 0
    MXNET = 1

@enum.unique
class ArrayType(enum.Enum):
    """Enumeration of types of arrays."""
    NUMPY = 0
    MXNET = 1

variants = {
        'numpy': (ArrayType.NUMPY, FunctionType.NUMPY),
        'mxnet': (ArrayType.MXNET, FunctionType.MXNET)
        }

allowed_types = {
        'numpy': numpy.allowed_types,
        'mxnet': mxnet.allowed_types
}
