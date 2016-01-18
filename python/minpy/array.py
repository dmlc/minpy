#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base type for arrays."""
from .utils import log
from .utils import common
import minpy.numpy as mnp

import numpy as np
import mxnet as mx

logger = log.get_logger(__name__)

class ArrayType(common.AutoNumber):
    """Enumeration of types of arrays."""
    NUMPY = ()
    MXNET = ()

class ArrayTypeMissingError(ValueError):
    pass
class UnknownArrayTypeError(ValueError):
    pass

class Array(object):
    """Base array type that provides convenient methods
    for arithmetic operations. The Array class is used for:
    1. Redirect all special member functions to corresponding pure function
    2. Redirect normal member functions to correct member functions of underlying
        array object

    Member:
        _data: A dict type { array_type : array_data }
    """
    __slots__ = ['_data']

    def __init__(self, data):
        self._data = data

    """ Return whether array data of given type exists in the underlying storage """
    def has_type(self, t):
        return t in self._data.keys()

    """ Get array data of given type. Raise exception if the type is missing """
    def get_data(self, t):
        if not self.has_type(t):
            raise ArrayTypeMissingError('Array data of type {} not found.'.format(t))
        return self._data[t]

    """ Create data of given type """
    def create_data(self, t):
        if not self.has_type(t):
            if t == ArrayType.NUMPY:
                mxarray = self.get_data(ArrayType.MXNET)
                self._data[ArrayType.NUMPY] = mxarray.asnumpy()
            elif t == ArrayType.MXNET:
                nparray = self.get_data(ArrayType.NUMPY)
                self._data[ArrayType.MXNET] = mx.nd.array(nparray)
            else:
                raise UnknownArrayTypeError('Unknown array type {}.'.format(t))

    @property
    def shape(self):
        return self._data.values()[0].shape

    def __neg__(self):
        return mnp.negate(self)

    def __add__(self, other):
        return mnp.add(self, other)

    def __sub__(self, other):
        return mnp.subtract(self, other)

    def __mul__(self, other):
        return mnp.multiply(self, other)

    def __div__(self, other):
        return mnp.divide(self, other)

    def __truediv__(self, other):
        return mnp.true_divide(self, other)

    def __pow__(self, other):
        return mnp.power(self, other)

    def __mod__(self, other):
        return mnp.mod(self, other)

    def __radd__(self, other):
        return mnp.add(other, self)

    def __rsub__(self, other):
        return mnp.subtract(other, self)

    def __rmul__(self, other):
        return mnp.multiply(other, self)

    def __rdiv__(self, other):
        return mnp.divide(other, self)

    def __rtruediv__(self, other):
        return mnp.true_divide(other, self)

    def __rpow__(self, other):
        return mnp.power(other, self)

    def __rmod__(self, other):
        return mnp.mod(other, self)
