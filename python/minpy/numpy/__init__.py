#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
from .mocking import Module
import numpy

sys.modules[__name__] = Module(numpy.__dict__)

class Delegate(object):
    __slots__ = ['_val']

    def __init__(self, val):
        self._val = val

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
