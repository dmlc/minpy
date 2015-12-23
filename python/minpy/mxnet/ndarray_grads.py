import operator
import mxnet as mx

from . import ndarray_wrapper as ndw
from . import random
from .. import core

class MXNetNDArrayNode(core.Node):
    def __init__(self, val):
        super(MXNetNDArrayNode, self).__init__(val)

    @property
    def shape(self):
        return self._val.shape

    def __neg__(self): return ndw.negative(self)

    def __add__(self, other): return ndw.add(     self, other)
    def __sub__(self, other): return ndw.subtract(self, other)
    def __mul__(self, other): return ndw.multiply(self, other)
    def __div__(self, other): return ndw.divide(  self, other)
    def __truediv__(self, other): return ndw.true_divide(self, other)
    def __pow__(self, other): return ndw.power   (self, other)
    def __mod__(self, other): return ndw.mod(     self, other)

    def __radd__(self, other): return ndw.add(     other, self)
    def __rsub__(self, other): return ndw.subtract(other, self)
    def __rmul__(self, other): return ndw.multiply(other, self)
    def __rdiv__(self, other): return ndw.divide(  other, self)
    def __rtruediv__(self, other): return ndw.true_divide(other, self)
    def __rpow__(self, other): return ndw.power(   other, self)
    def __rmod__(self, other): return ndw.mod(     other, self)

