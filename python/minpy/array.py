#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base type for arrays."""
from .utils import log
from .utils import common
import minpy.numpy
import mxnet
import numpy

logger = log.get_logger(__name__)


class ArrayType(common.AutoNumber):
    """Enumeration of types of arrays."""
    NUMPY = ()
    MXNET = ()


def get_array_type(arr: Union[numpy.ndarray, mxnet.narray.NArray]) -> ArrayType:
    t = type(arr)
    if t == numpy.ndarray:
        return ArrayType.NUMPY
    elif t == mxnet.nd.NArray:
        return ArrayType.MXNET
    else:
        raise UnknownArrayTypeError(
            'Array data of type {} unknown.'.format(t))


def get_real_type(arr: ArrayType) -> type:
    if arr == ArrayType.NUMPY:
        return numpy.ndarray
    elif arr == ArrayType.MXNET:
        return mxnet.nd.NArray
    else:
        raise UnknownArrayTypeError(
            'Array data of type {} unknown.'.format(arr))


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
    _node = Node()  # TODO derivative info
    _data = dict()  # TODO real data

    def __init__(self, data):
        self._data = data
        # TODO assert data as either numpy.array or mxnet.ndarray, or throw
        # exception

    """ Return whether array data of given type exists in the underlying storage """

    def has_type(self, t):
        return t in self._data.keys()

    """ Get array data of given type. Raise exception if the type is missing """

    def get_data(self, t):
        if t not in self._data:
            raise ArrayTypeMissingError(
                'Array data of type {} not found.'.format(t))
        return self._data[t]

    """ Create data of given type """

    def create_data(self, t):
        if t not in self._data:
            if t == ArrayType.NUMPY:
                mxarray = self.get_data(ArrayType.MXNET)
                self._data[ArrayType.NUMPY] = mxarray.asnumpy()
            elif t == ArrayType.MXNET:
                nparray = self.get_data(ArrayType.NUMPY)
                self._data[ArrayType.MXNET] = mxnet.nd.array(nparray)
            else:
                raise UnknownArrayTypeError('Unknown array type {}.'.format(t))

    @property
    def shape(self):
        return self._data.values()[0].shape

    # TODO special function redirection and __getattr__ redirection
    def __neg__(self):
        return minpy.numpy.negate(self)

    def __add__(self, other):
        return minpy.numpy.add(self, other)

    def __sub__(self, other):
        return minpy.numpy.subtract(self, other)

    def __mul__(self, other):
        return minpy.numpy.multiply(self, other)

    def __div__(self, other):
        return minpy.numpy.divide(self, other)

    def __truediv__(self, other):
        return minpy.numpy.true_divide(self, other)

    def __pow__(self, other):
        return minpy.numpy.power(self, other)

    def __mod__(self, other):
        return minpy.numpy.mod(self, other)

    def __radd__(self, other):
        return minpy.numpy.add(other, self)

    def __rsub__(self, other):
        return minpy.numpy.subtract(other, self)

    def __rmul__(self, other):
        return minpy.numpy.multiply(other, self)

    def __rdiv__(self, other):
        return minpy.numpy.divide(other, self)

    def __rtruediv__(self, other):
        return minpy.numpy.true_divide(other, self)

    def __rpow__(self, other):
        return minpy.numpy.power(other, self)

    def __rmod__(self, other):
        return minpy.numpy.mod(other, self)
