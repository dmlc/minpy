#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base type for arrays."""
from .utils import log
from .utils import common
from . import core
import typing
import minpy.numpy
import mxnet
import numpy

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
    """Base array type.

    It provides convenient methods for arithmetic operations. The Array class
    is used for:
    1. Redirect all special member functions to corresponding pure function.
    2. Redirect normal member functions to correct member functions of
    underlying array object.
    """
    _node = core.Node()  # TODO derivative info
    _data = dict()  # TODO real data

    @staticmethod
    def to_array_type(arr: typing.Union[numpy.ndarray, mxnet.narray.NArray]
                      ) -> ArrayType:
        t = type(arr)
        if t == numpy.ndarray:
            return ArrayType.NUMPY
        elif t == mxnet.nd.NArray:
            return ArrayType.MXNET
        else:
            raise UnknownArrayTypeError(
                'Array data of type {} unknown.'.format(t))

    @staticmethod
    def to_real_type(arr: ArrayType) -> type:
        if arr == ArrayType.NUMPY:
            return numpy.ndarray
        elif arr == ArrayType.MXNET:
            return mxnet.nd.NArray
        else:
            raise UnknownArrayTypeError(
                'Array data of type {} unknown.'.format(arr))

    def __init__(self, data):
        t = Array.to_array_type(data)
        self._data[t] = data

    def has_type(self, t):
        """Return whether array data of given type exists in the underlying storage.
        """
        return t in self._data.keys()

    def get_data(self, t):
        """Get array data of given type. Raise exception if the type is missing.
        """
        if t not in self._data:
            raise ArrayTypeMissingError(
                'Array data of type {} not found.'.format(t))
        return self._data[t]

    def create_data(self, t: ArrayType):
        """Create data of given type."""
        if t not in self._data:
            if t == ArrayType.NUMPY:
                mxarray = self.get_data(ArrayType.MXNET)
                # TODO conversion
                self._data[ArrayType.NUMPY] = mxarray.asnumpy()
            elif t == ArrayType.MXNET:
                nparray = self.get_data(ArrayType.NUMPY)
                self._data[ArrayType.MXNET] = mxnet.nd.array(nparray)
            else:
                raise UnknownArrayTypeError(
                    'Array data of type {} unknown.'.format(t))

    @property
    def shape(self):
        return self._data.values()[0].shape

    # TODO special function redirection and __getattr__ redirection

    def __getattr__(self, name):
        # TODO la magie
        pass

    def __cmp__(self, other):
        pass

    def __eq__(self, other):
        pass

    def __ne__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __ge__(self, other):
        pass

    def __pos__(self):
        pass

    def __neg__(self):
        return minpy.numpy.negate(self)

    def __abs__(self):
        pass

    def __invert__(self):
        pass

    def __round__(self, n):
        pass

    def __floor__(self):
        pass

    def __ceil__(self):
        pass

    def __trunc__(self):
        pass

    def __add__(self, other):
        return minpy.numpy.add(self, other)

    def __sub__(self, other):
        return minpy.numpy.subtract(self, other)

    def __mul__(self, other):
        return minpy.numpy.multiply(self, other)

    def __floordiv__(self, other):
        pass

    def __div__(self, other):
        return minpy.numpy.divide(self, other)

    def __truediv__(self, other):
        return minpy.numpy.true_divide(self, other)

    def __mod__(self, other):
        return minpy.numpy.mod(self, other)

    def __divmod__(self, other):
        pass

    def __pow__(self, other):
        return minpy.numpy.power(self, other)

    def __lshift__(self, other):
        pass

    def __rshift__(self, other):
        pass

    def __and__(self, other):
        pass

    def __or__(self, other):
        pass

    def __xor__(self, other):
        pass

    def __radd__(self, other):
        return minpy.numpy.add(other, self)

    def __rsub__(self, other):
        return minpy.numpy.subtract(other, self)

    def __rmul__(self, other):
        return minpy.numpy.multiply(other, self)

    def __rfloordiv__(self, other):
        pass

    def __rdiv__(self, other):
        return minpy.numpy.divide(other, self)

    def __rtruediv__(self, other):
        return minpy.numpy.true_divide(other, self)

    def __rmod__(self, other):
        return minpy.numpy.mod(other, self)

    def __rdivmod__(self, other):
        return minpy.numpy.mod(other, self)

    def __rpow__(self, other):
        return minpy.numpy.power(other, self)

    def __rlshift__(self, other):
        pass

    def __rrshift__(self, other):
        pass

    def __rand__(self, other):
        pass

    def __ror__(self, other):
        pass

    def __rxor__(self, other):
        pass

    def __iadd__(self, other):
        return minpy.numpy.add(other, self)

    def __isub__(self, other):
        return minpy.numpy.subtract(other, self)

    def __imul__(self, other):
        return minpy.numpy.multiply(other, self)

    def __ifloordiv__(self, other):
        pass

    def __idiv__(self, other):
        return minpy.numpy.divide(other, self)

    def __itruediv__(self, other):
        return minpy.numpy.true_divide(other, self)

    def __imod__(self, other):
        return minpy.numpy.mod(other, self)

    def __ipow__(self, other):
        return minpy.numpy.power(other, self)

    def __ilshift__(self, other):
        pass

    def __irshift__(self, other):
        pass

    def __iand__(self, other):
        pass

    def __ior__(self, other):
        pass

    def __ixor__(self, other):
        pass
