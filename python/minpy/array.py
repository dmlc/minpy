#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base type for arrays."""
from .utils import log
from .utils import common

logger = log.get_logger(__name__)


class ArrayType(common.AutoNumber):
    """Enumeration of types of arrays."""
    NUMPY = ()
    MXNET = ()


class Array(object):
    """Base array type that provides convenient methods
    for arithmetic operations.
    """
    __slots__ = ['_type', '_data']

    def __init__(self, data, t):
        self._type = t
        self._data = data

    def get_type(self):
        return self._type

    def convert(self, to):
        # TODO implement
        self._type = to
        return self

    @property
    def shape(self):
        pass

    def __neg__(self):
        pass

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __div__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __pow__(self, other):
        pass

    def __mod__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __rdiv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __rpow__(self, other):
        pass

    def __rmod__(self, other):
        pass
