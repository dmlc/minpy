#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable= invalid-name
""" Package for different implementations of array computations """
from __future__ import absolute_import
from __future__ import print_function

from minpy.array_variants import numpy
from minpy.array_variants import mxnet
from minpy.utils import common

class ArrayType(object):
    """Enumeration of types of arrays."""
    # pylint: disable=no-init
    NUMPY = 0
    MXNET = 1

variants = {'numpy': ArrayType.NUMPY, 'mxnet': ArrayType.MXNET}
array_types = {'numpy': numpy.array_type, 'mxnet': mxnet.array_type}
variants_repr = {ArrayType.NUMPY: 'NumPy', ArrayType.MXNET: 'MXNet'}
number_types = {'native': [int, float], 'numpy': numpy.number_type, 'mxnet': mxnet.number_type}
