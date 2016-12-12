#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable=invalid-name, no-member
""" Package for numpy array implementations """
from __future__ import absolute_import

import numpy
from minpy.array_variants.numpy import numpy_core

array_type = numpy.ndarray
number_type = [
    numpy.float, numpy.float16, numpy.float32, numpy.float64, numpy.int,
    numpy.int32, numpy.int64
]

register_primitives = numpy_core.register_primitives
def_grads = numpy_core.def_grads
