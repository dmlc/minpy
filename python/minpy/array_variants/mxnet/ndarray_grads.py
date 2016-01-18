import operator
import mxnet.ndarray as nd

from . import ndarray_wrapper as ndw
from .. import core
from .. import array


def identity(x):
    return x


def unbroadcast(ans, x, gradfun):
    # TODO currently no broadcasting for mx.ndarray
    return gradfun

# dot
ndw.dot.def_grad(lambda ans, a, b: lambda g: nd.dot(g, b.T))
ndw.dot.def_grad(lambda ans, a, b: lambda g: nd.dot(a.T, g), argnum=1)
# non-linear
#ndw.tanh.def_grad(lambda ans, x: lambda g: g / np.cosh(x) ** 2)
ndw.exp.def_grad(lambda ans, x: lambda g: g * ans)
ndw.log.def_grad(lambda ans, x: lambda g: g / x)
# reduce
ndw.sum.def_grad(lambda ans, x: lambda g: nd.full(x.shape, g, x.context))
# + - * /
ndw.multiply.def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g * y))
ndw.multiply.def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: x * g), argnum=1)
ndw.add.def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
ndw.add.def_grad(lambda ans, x, y: unbroadcast(ans, y, identity), argnum=1)
ndw.subtract.def_grad(lambda ans, x, y: unbroadcast(ans, x, identity))
ndw.subtract.def_grad(lambda ans, x, y: unbroadcast(ans, y, operator.neg), argnum=1)
ndw.divide.def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
ndw.divide.def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: - g * x / y * y), argnum=1)
ndw.true_divide.def_grad(lambda ans, x, y: unbroadcast(ans, x, lambda g: g / y))
ndw.true_divide.def_grad(lambda ans, x, y: unbroadcast(ans, y, lambda g: - g * x / y * y), argnum=1)
# power
#ndw.power.def_grad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * y * x ** (y - 1)))
#ndw.power.def_grad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * nd.log(x) * x ** y), argnum=1)
# mod
#ndw.mod.def_grad(lambda ans, x, y : unbroadcast(ans, x, identity))
#ndw.mod.def_grad(lambda ans, x, y : unbroadcast(ans, y, lambda g : - g * nd.floor(x/y)), argnum=1)
# negate
ndw.negative.def_grad(lambda ans, x: operator.neg)


class MXNetNDArrayNode(array.Array):
    def __init__(self, val):
        super(MXNetNDArrayNode, self).__init__(val)

    @property
    def shape(self):
        return self._val.shape

    def asnumpy(self):
        return self._val.asnumpy()

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

core.register_node_type(nd.NDArray, MXNetNDArrayNode)
