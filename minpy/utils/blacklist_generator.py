from __future__ import division
from math import pi
import minpy
import minpy.numpy as np
from minpy.dispatch.policy import AutoBlacklistPolicy
import logging


def test_ufunc():
    x = np.array([-1.2, 1.2])
    np.absolute(x)
    np.absolute(1.2 + 1j)
    x = np.linspace(start=-10, stop=10, num=101)
    np.add(1.0, 4.0)
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)
    np.add(x1, x2)
    np.arccos([1, -1])
    x = np.linspace(-1, 1, num=100)
    np.arccosh([np.e, 10.0])
    np.arccosh(1)
    np.arcsin(0)
    np.arcsinh(np.array([np.e, 10.0]))
    np.arctan([0, 1])
    np.pi/4
    x = np.linspace(-10, 10)
    x = np.array([-1, +1, +1, -1])
    y = np.array([-1, -1, +1, +1])
    np.arctan2(y, x) * 180 / np.pi
    np.arctan2([1., -1.], [0., 0.])
    np.arctan2([0., 0., np.inf], [+0., -0., np.inf])
    np.arctanh([0, -0.5])
    np.bitwise_and(13, 17)
    np.bitwise_and(14, 13)
    # np.binary_repr(12)    return str
    np.bitwise_and([14,3], 13)
    np.bitwise_and([11,7], [4,25])
    np.bitwise_and(np.array([2,5,255]), np.array([3,14,16]))
    np.bitwise_and([True, True], [False, True])
    np.bitwise_or(13, 16)
    # np.binary_repr(29)
    np.bitwise_or(32, 2)
    np.bitwise_or([33, 4], 1)
    np.bitwise_or([33, 4], [1, 2])
    np.bitwise_or(np.array([2, 5, 255]), np.array([4, 4, 4]))
    # np.array([2, 5, 255]) | np.array([4, 4, 4])
    np.bitwise_or(np.array([2, 5, 255, 2147483647L], dtype=np.int32),
                  np.array([4, 4, 4, 2147483647L], dtype=np.int32))
    np.bitwise_or([True, True], [False, True])
    np.bitwise_xor(13, 17)
    # np.binary_repr(28)
    np.bitwise_xor(31, 5)
    np.bitwise_xor([31,3], 5)
    np.bitwise_xor([31,3], [5,6])
    np.bitwise_xor([True, True], [False, True])
    a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    np.ceil(a)
    a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    np.trunc(a)
    np.cos(np.array([0, np.pi/2, np.pi]))
    np.cosh(0)
    x = np.linspace(-4, 4, 1000)
    rad = np.arange(12.)*np.pi/6
    np.degrees(rad)
    out = np.zeros((rad.shape))
    r = np.degrees(rad, out)
    # np.all(r == out) return bool
    np.rad2deg(np.pi/2)
    np.divide(2.0, 4.0)
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)
    np.divide(2, 4)
    np.divide(2, 4.)
    np.equal([0, 1, 3], np.arange(3))
    np.equal(1, np.ones(1))
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    np.exp2([2, 3])
    np.expm1(1e-10)
    np.exp(1e-10) - 1
    np.fabs(-1)
    np.fabs([-1.2, 1.2])
    a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    np.floor(a)
    np.floor_divide(7,3)
    np.floor_divide([1., 2., 3., 4.], 2.5)
    np.fmod([-3, -2, -1, 1, 2, 3], 2)
    np.remainder([-3, -2, -1, 1, 2, 3], 2)
    np.fmod([5, 3], [2, 2.])
    a = np.arange(-3, 3).reshape(3, 2)
    np.fmod(a, [2,2])
    np.greater([4,2],[2,2])
    a = np.array([4,2])
    b = np.array([2,2])
    a > b
    np.greater_equal([4, 2, 1], [2, 2, 2])
    np.hypot(3*np.ones((3, 3)), 4*np.ones((3, 3)))
    np.hypot(3*np.ones((3, 3)), [4])
    np.bitwise_not is np.invert
    np.invert(np.array([13], dtype=np.uint8))
    # np.binary_repr(242, width=8)
    np.invert(np.array([13], dtype=np.uint16))
    np.invert(np.array([13], dtype=np.int8))
    # np.binary_repr(-14, width=8)
    np.invert(np.array([True, False]))
    # np.isfinite(1)
    # np.isfinite(0)
    # np.isfinite(np.nan)
    # np.isfinite(np.inf)
    # np.isfinite(np.NINF)
    x = np.array([-np.inf, 0., np.inf])
    y = np.array([2, 2, 2])
    np.isfinite(x, y)
    # np.isinf(np.inf)
    # np.isinf(np.nan)
    # np.isinf(np.NINF)
    # np.isinf([np.inf, -np.inf, 1.0, np.nan])
    x = np.array([-np.inf, 0., np.inf])
    y = np.array([2, 2, 2])
    # np.isinf(x, y)
    # np.isnan(np.nan)
    # np.isnan(np.inf)
    # np.binary_repr(5)
    np.left_shift(5, 2)
    # np.binary_repr(20)
    np.left_shift(5, [1,2,3])
    np.less([1, 2], [2, 2])
    np.less_equal([4, 2, 1], [2, 2, 2])
    x = np.array([0, 1, 2, 2**4])
    xi = np.array([0+1.j, 1, 2+0.j, 4.j])
    np.log2(xi)
    prob1 = np.log(1e-50)
    prob2 = np.log(2.5e-50)
    prob12 = np.logaddexp(prob1, prob2)
    prob12
    np.exp(prob12)
    prob1 = np.log2(1e-50)
    prob2 = np.log2(2.5e-50)
    prob12 = np.logaddexp2(prob1, prob2)
    prob1, prob2, prob12
    2**prob12
    np.log1p(1e-99)
    np.log(1 + 1e-99)
    # np.logical_and(True, False)
    # np.logical_and([True, False], [False, False])
    x = np.arange(5)
    # np.logical_and(x>1, x<4)
    # np.logical_not(3)
    # np.logical_not([True, False, 0, 1])
    x = np.arange(5)
    # np.logical_not(x<3)
    # np.logical_or(True, False)
    # np.logical_or([True, False], [False, False])
    x = np.arange(5)
    # np.logical_or(x < 1, x > 3)
    # np.logical_xor(True, False)
    # np.logical_xor([True, True, False, False], [True, False, True, False])
    x = np.arange(5)
    # np.logical_xor(x < 1, x > 3)
    # np.logical_xor(0, np.eye(2))
    np.maximum([2, 3, 4], [1, 5, 2])
    # np.maximum([np.nan, 0, np.nan], [0, np.nan, np.nan])
    # np.maximum(np.Inf, 1)
    np.minimum([2, 3, 4], [1, 5, 2])
    # np.minimum([np.nan, 0, np.nan],[0, np.nan, np.nan])
    # np.minimum(-np.Inf, 1)
    np.fmax([2, 3, 4], [1, 5, 2])
    np.fmax(np.eye(2), [0.5, 2])
    # np.fmax([np.nan, 0, np.nan],[0, np.nan, np.nan])
    np.fmin([2, 3, 4], [1, 5, 2])
    np.fmin(np.eye(2), [0.5, 2])
    # np.fmin([np.nan, 0, np.nan],[0, np.nan, np.nan])
    np.modf([0, 3.5])
    np.modf(-0.5)
    np.multiply(2.0, 4.0)
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)
    np.multiply(x1, x2)
    np.negative([1.,-1.])
    np.not_equal([1.,2.], [1., 3.])
    np.not_equal([1, 2], [[1, 3],[1, 4]])
    x1 = range(6)
    np.power(x1, 3)
    x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    np.power(x1, x2)
    x2 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    np.power(x1, x2)
    deg = np.arange(12.) * 30.
    np.radians(deg)
    out = np.zeros((deg.shape))
    ret = np.radians(deg, out)
    ret is out
    np.deg2rad(180)
    np.reciprocal(2.)
    np.reciprocal([1, 2., 3.33])
    np.remainder([4, 7], [2, 3])
    np.remainder(np.arange(7), 5)
    # np.binary_repr(10)
    np.right_shift(10, 1)
    # np.binary_repr(5)
    np.right_shift(10, [1,2,3])
    a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    np.rint(a)
    np.sign([-5., 4.5])
    np.sign(0)
    # np.sign(5-2j)
    # np.signbit(-1.2)
    np.signbit(np.array([1, -2.3, 2.1]))
    np.copysign(1.3, -1)
    np.copysign([-1, 0, 1], -1.1)
    np.copysign([-1, 0, 1], np.arange(3)-1)
    np.sin(np.pi/2.)
    np.sin(np.array((0., 30., 45., 60., 90.)) * np.pi / 180. )
    x = np.linspace(-np.pi, np.pi, 201)
    np.sinh(0)
    # np.sinh(np.pi*1j/2)
    np.sqrt([1,4,9])
    np.sqrt([4, -1, -3+4J])
    np.cbrt([1,8,27])
    np.square([-1j, 1])
    np.subtract(1.0, 4.0)
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)
    np.subtract(x1, x2)
    np.tan(np.array([-pi,pi/2,pi]))
    np.tanh((0, np.pi*1j, np.pi*1j/2))
    x = np.arange(5)
    np.true_divide(x, 4)
    x = np.arange(9)
    y1, y2 = np.frexp(x)
    y1 * 2**y2
    np.ldexp(5, np.arange(4))
    x = np.arange(6)
    np.ldexp(*np.frexp(x))


def test_numeric():
    # 'newaxis', 'ndarray', 'flatiter', 'nditer', 'nested_iters', 'ufunc',
    # 'arange', 'array', 'zeros', 'count_nonzero', 'empty', 'broadcast',
    # 'dtype', 'fromstring', 'fromfile', 'frombuffer', 'int_asbuffer',
    # 'where', 'argwhere', 'copyto', 'concatenate', 'fastCopyAndTranspose',
    # 'lexsort', 'set_numeric_ops', 'can_cast', 'promote_types',
    # 'min_scalar_type', 'result_type', 'asarray', 'asanyarray',
    # 'ascontiguousarray', 'asfortranarray', 'isfortran', 'empty_like',
    # 'zeros_like', 'ones_like', 'correlate', 'convolve', 'inner', 'dot',
    # 'einsum', 'outer', 'vdot', 'alterdot', 'restoredot', 'roll',
    # 'rollaxis', 'moveaxis', 'cross', 'tensordot', 'array2string',
    # 'get_printoptions', 'set_printoptions', 'array_repr', 'array_str',
    # 'set_string_function', 'little_endian', 'require', 'fromiter',
    # 'array_equal', 'array_equiv', 'indices', 'fromfunction', 'isclose', 'load',
    # 'loads', 'isscalar', 'binary_repr', 'base_repr', 'ones', 'identity',
    # 'allclose', 'compare_chararrays', 'putmask', 'seterr', 'geterr',
    # 'setbufsize', 'getbufsize', 'seterrcall', 'geterrcall', 'errstate',
    # 'flatnonzero', 'Inf', 'inf', 'infty', 'Infinity', 'nan', 'NaN', 'False_',
    # 'True_', 'bitwise_not', 'full', 'full_like', 'matmul'
    x = np.arange(6)
    x = x.reshape((2, 3))
    np.zeros_like(x)
    y = np.arange(3, dtype=np.float)
    np.zeros_like(y)
    np.ones(5)
    np.ones((5,), dtype=np.int)
    np.ones((2, 1))
    s = (2,2)
    np.ones(s)
    x = np.arange(6)
    x = x.reshape((2, 3))
    np.ones_like(x)
    y = np.arange(3, dtype=np.float)
    np.ones_like(y)
    np.full((2, 2), np.inf)
    x = np.arange(6, dtype=np.int)
    np.full_like(x, 1)
    np.full_like(x, 0.1)
    np.full_like(y, 0.1)
    np.count_nonzero(np.eye(4))
    np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]])
    np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=0)
    np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1)
    a = [1, 2]
    np.asarray(a)
    a = np.array([1, 2])
    np.asarray(a) is a
    a = np.array([1, 2], dtype=np.float32)
    np.asarray(a, dtype=np.float32) is a
    np.asarray(a, dtype=np.float64) is a
    np.asarray(a) is a
    np.asanyarray(a) is a
    a = [1, 2]
    np.asanyarray(a)
    np.asanyarray(a) is a
    x = np.arange(6).reshape(2,3)
    np.ascontiguousarray(x, dtype=np.float32)
    x = np.arange(6).reshape(2,3)
    y = np.asfortranarray(x)
    x = np.arange(6).reshape(2,3)
    y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F'])
    a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    np.isfortran(a)
    b = np.array([[1, 2, 3], [4, 5, 6]], order='FORTRAN')
    np.isfortran(b)
    a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    np.isfortran(a)
    b = a.T
    np.isfortran(b)
    np.isfortran(np.array([1, 2], order='FORTRAN'))
    x = np.arange(6).reshape(2,3)
    np.argwhere(x>1)
    x = np.arange(-2, 3)
    np.flatnonzero(x)
    np.correlate([1, 2, 3], [0, 1, 0.5])
    np.correlate([1, 2, 3], [0, 1, 0.5], "same")
    np.correlate([1, 2, 3], [0, 1, 0.5], "full")
    np.correlate([1+1j, 2, 3-1j], [0, 1, 0.5j], 'full')
    np.correlate([0, 1, 0.5j], [1+1j, 2, 3-1j], 'full')
    np.convolve([1, 2, 3], [0, 1, 0.5])
    np.convolve([1,2,3],[0,1,0.5], 'same')
    np.convolve([1,2,3],[0,1,0.5], 'valid')
    rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    # im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
    # grid = rl + im
    x = np.array(['a', 'b', 'c'], dtype=object)
    np.outer(x, [1, 2, 3])
    a = np.arange(60.).reshape(3,4,5)
    b = np.arange(24.).reshape(4,3,2)
    c = np.tensordot(a,b, axes=([1,0],[0,1]))
    c.shape
    # A slower but equivalent way of computing the same...
    d = np.zeros((5,2))
    a = np.array(range(1, 9))
    A = np.array(('a', 'b', 'c', 'd'), dtype=object)
    x = np.arange(10)
    np.roll(x, 2)
    x2 = np.reshape(x, (2,5))
    np.roll(x2, 1)
    np.roll(x2, 1, axis=0)
    np.roll(x2, 1, axis=1)
    a = np.ones((3,4,5,6))
    np.rollaxis(a, 3, 1).shape
    np.rollaxis(a, 2).shape
    np.rollaxis(a, 1, 4).shape
    x = np.zeros((3, 4, 5))
    np.moveaxis(x, 0, -1).shape
    np.moveaxis(x, -1, 0).shape
    np.transpose(x).shape
    np.moveaxis(x, [0, 1], [-1, -2]).shape
    np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    x = [1, 2, 3]
    y = [4, 5, 6]
    np.cross(x, y)
    x = [1, 2]
    y = [4, 5, 6]
    np.cross(x, y)
    x = [1, 2, 0]
    y = [4, 5, 6]
    np.cross(x, y)
    x = [1,2]
    y = [4,5]
    np.cross(x, y)
    x = np.array([[1,2,3], [4,5,6]])
    y = np.array([[4,5,6], [1,2,3]])
    np.cross(x, y)
    np.cross(x, y, axisc=0)
    x = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
    y = np.array([[7, 8, 9], [4,5,6], [1,2,3]])
    np.cross(x, y)
    np.cross(x, y, axisa=0, axisb=0)
    # np.array_repr(np.array([1,2]))
    # np.array_repr(np.ma.array([0.]))
    # np.array_repr(np.array([], np.int32))
    x = np.array([1e-6, 4e-7, 2, 3])
    # np.array_repr(x, precision=6, suppress_small=True)
    # np.array_str(np.arange(3))
    a = np.arange(10)
    x = np.arange(4)
    np.set_string_function(lambda x:'random', repr=False)
    grid = np.indices((2, 3))
    grid.shape
    grid[0]        # row indices
    grid[1]        # column indices
    x = np.arange(20).reshape(5, 4)
    row, col = np.indices((2, 3))
    x[row, col]
    np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
    np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
    np.isscalar(3.1)
    np.isscalar([3.1])
    np.isscalar(False)
    # np.binary_repr(3)
    # np.binary_repr(-3)
    # np.binary_repr(3, width=4)
    # np.binary_repr(-3, width=3)
    # np.binary_repr(-3, width=5)
    # np.base_repr(5)
    # np.base_repr(6, 5)
    # np.base_repr(7, base=5, padding=3)
    # np.base_repr(10, base=16)
    # np.base_repr(32, base=16)
    np.identity(3)
    np.allclose([1e10,1e-7], [1.00001e10,1e-8])
    np.allclose([1e10,1e-8], [1.00001e10,1e-9])
    np.allclose([1e10,1e-8], [1.0001e10,1e-9])
    # np.allclose([1.0, np.nan], [1.0, np.nan])
    # np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    np.isclose([1e10,1e-7], [1.00001e10,1e-8])
    np.isclose([1e10,1e-8], [1.00001e10,1e-9])
    np.isclose([1e10,1e-8], [1.0001e10,1e-9])
    # np.isclose([1.0, np.nan], [1.0, np.nan])
    # np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    np.array_equal([1, 2], [1, 2])
    np.array_equal(np.array([1, 2]), np.array([1, 2]))
    np.array_equal([1, 2], [1, 2, 3])
    np.array_equal([1, 2], [1, 4])
    np.array_equiv([1, 2], [1, 2])
    np.array_equiv([1, 2], [1, 3])
    np.array_equiv([1, 2], [[1, 2], [1, 2]])
    np.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]])
    np.array_equiv([1, 2], [[1, 2], [1, 3]])


def test_fromnumeric():
    # Functions
    # 'alen', 'all', 'alltrue', 'amax', 'amin', 'any', 'argmax',
    # 'argmin', 'argpartition', 'argsort', 'around', 'choose', 'clip',
    # 'compress', 'cumprod', 'cumproduct', 'cumsum', 'diagonal', 'mean',
    # 'ndim', 'nonzero', 'partition', 'prod', 'product', 'ptp', 'put',
    # 'rank', 'ravel', 'repeat', 'reshape', 'resize', 'round_',
    # 'searchsorted', 'shape', 'size', 'sometrue', 'sort', 'squeeze',
    # 'std', 'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var',
    a = [4, 3, 5, 7, 6, 8]
    indices = [0, 1, 4]
    np.take(a, indices)
    a = np.array(a)
    # a[indices]
    np.take(a, [[0, 1], [2, 3]])
    a = np.zeros((10, 2))
    b = a.T
    a = np.arange(6).reshape((3, 2))
    np.reshape(a, (2, 3)) # C-like index ordering
    np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
    np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
    a = np.array([[1,2,3], [4,5,6]])
    np.reshape(a, 6)
    np.reshape(a, 6, order='F')
    np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    choices = [[0, 1, 2, 3], [10, 11, 12, 13],
               [20, 21, 22, 23], [30, 31, 32, 33]]
    np.choose([2, 3, 1, 0], choices)
    np.choose([2, 4, 1, 0], choices, mode='clip') # 4 goes to 3 (4-1)
    np.choose([2, 4, 1, 0], choices, mode='wrap') # 4 goes to (4 mod 4)
    a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    choices = [-10, 10]
    np.choose(a, choices)
    a = np.array([0, 1]).reshape((2,1,1))
    c1 = np.array([1, 2, 3]).reshape((1,3,1))
    c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
    np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2
    np.repeat(3, 4)
    x = np.array([[1,2],[3,4]])
    np.repeat(x, 2)
    np.repeat(x, 3, axis=1)
    np.repeat(x, [1, 2], axis=0)
    a = np.arange(5)
    np.put(a, [0, 2], [-44, -55])
    a = np.arange(5)
    np.put(a, 22, -5, mode='clip')
    x = np.array([[1,2,3]])
    np.swapaxes(x,0,1)
    x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    np.swapaxes(x,0,2)
    x = np.arange(4).reshape((2,2))
    np.transpose(x)
    x = np.ones((1, 2, 3))
    np.transpose(x, (1, 0, 2)).shape
    a = np.array([3, 4, 2, 1])
    np.partition(a, 3)
    np.partition(a, (1, 3))
    x = np.array([3, 4, 2, 1])
    x[np.argpartition(x, 3)]
    x[np.argpartition(x, (1, 3))]
    x = [3, 4, 2, 1]
    np.array(x)[np.argpartition(x, 3)]
    a = np.array([[1,4],[3,1]])
    np.sort(a)                # sort along the last axis
    np.sort(a, axis=None)     # sort the flattened array
    np.sort(a, axis=0)        # sort along the first axis
    dtype = [('name', 'S10'), ('height', float), ('age', int)]
    values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
              ('Galahad', 1.7, 38)]
    a = np.array(values, dtype=dtype)       # create a structured array
    np.sort(a, order='height')                        # doctest: +SKIP
    np.sort(a, order=['age', 'height'])               # doctest: +SKIP
    x = np.array([3, 1, 2])
    np.argsort(x)
    x = np.array([[0, 3], [2, 2]])
    np.argsort(x, axis=0)
    np.argsort(x, axis=1)
    x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
    np.argsort(x, order=('x','y'))
    np.argsort(x, order=('y','x'))
    a = np.arange(6).reshape(2,3)
    np.argmax(a)
    np.argmax(a, axis=0)
    np.argmax(a, axis=1)
    b = np.arange(6)
    b[1] = 5
    np.argmax(b) # Only the first occurrence is returned.
    a = np.arange(6).reshape(2,3)
    np.argmin(a)
    np.argmin(a, axis=0)
    np.argmin(a, axis=1)
    b = np.arange(6)
    b[4] = 0
    np.argmin(b) # Only the first occurrence is returned.
    np.searchsorted([1,2,3,4,5], 3)
    np.searchsorted([1,2,3,4,5], 3, side='right')
    np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
    a=np.array([[0,1],[2,3]])
    np.resize(a,(2,3))
    np.resize(a,(1,4))
    np.resize(a,(2,4))
    x = np.array([[[0], [1], [2]]])
    x.shape
    np.squeeze(x).shape
    np.squeeze(x, axis=(2,)).shape
    a = np.arange(4).reshape(2,2)
    a = np.arange(8).reshape(2,2,2); a
    a[:,:,0] # main diagonal is [0 6]
    a[:,:,1] # main diagonal is [1 7]
    np.trace(np.eye(3))
    a = np.arange(8).reshape((2,2,2))
    np.trace(a)
    a = np.arange(24).reshape((2,2,2,3))
    np.trace(a).shape
    x = np.array([[1, 2, 3], [4, 5, 6]])
    np.ravel(x)
    x.reshape(-1)
    np.ravel(x, order='F')
    np.ravel(x.T)
    np.ravel(x.T, order='A')
    a = np.arange(3)[::-1]; a
    # a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
    x = np.eye(3)
    np.nonzero(x)
    x[np.nonzero(x)]
    np.transpose(np.nonzero(x))
    a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    a > 3
    np.nonzero(a > 3)
    np.shape(np.eye(3))
    np.shape([[1, 2]])
    np.shape([0])
    np.shape(0)
    a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
    np.shape(a)
    a.shape
    a = np.array([[1, 2], [3, 4], [5, 6]])
    np.compress([0, 1], a, axis=0)
    np.compress([False, True, True], a, axis=0)
    np.compress([False, True], a, axis=1)
    np.compress([False, True], a)
    a = np.arange(10)
    np.clip(a, 1, 8)
    np.clip(a, 3, 6, out=a)
    a = np.arange(10)
    np.clip(a, [3,4,1,1,1,4,4,4,4,4], 8)
    np.sum([])
    np.sum([0.5, 1.5])
    np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    np.sum([[0, 1], [0, 5]])
    np.sum([[0, 1], [0, 5]], axis=0)
    np.sum([[0, 1], [0, 5]], axis=1)
    # np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    # np.any([[True, False], [True, True]])
    # np.any([[True, False], [False, False]], axis=0)
    # np.any([-1, 0, 5])
    # np.any(np.nan)
    # np.all([[True,False],[True,True]])
    # np.all([[True,False],[True,True]], axis=0)
    # np.all([-1, 4, 5])
    # np.all([1.0, np.nan])
    a = np.array([[1,2,3], [4,5,6]])
    np.cumsum(a)
    np.cumsum(a, dtype=float)     # specifies type of output value(s)
    np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
    np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
    x = np.arange(4).reshape((2,2))
    np.ptp(x, axis=0)
    np.ptp(x, axis=1)
    a = np.arange(4).reshape((2,2))
    np.amax(a)           # Maximum of the flattened array
    np.amax(a, axis=0)   # Maxima along the first axis
    np.amax(a, axis=1)   # Maxima along the second axis
    b = np.arange(5, dtype=np.float)
    # b[2] = np.NaN
    np.amax(b)
    np.nanmax(b)
    a = np.arange(4).reshape((2,2))
    np.amin(a)           # Minimum of the flattened array
    np.amin(a, axis=0)   # Minima along the first axis
    np.amin(a, axis=1)   # Minima along the second axis
    b = np.arange(5, dtype=np.float)
    # b[2] = np.NaN
    np.amin(b)
    np.nanmin(b)
    a = np.zeros((7,4,5))
    a.shape[0]
    np.alen(a)
    x = np.array([536870910, 536870910, 536870910, 536870910])
    np.prod(x) #random
    np.prod([])
    np.prod([1.,2.])
    np.prod([[1.,2.],[3.,4.]])
    np.prod([[1.,2.],[3.,4.]], axis=1)
    x = np.array([1, 2, 3], dtype=np.uint8)
    # np.prod(x).dtype == np.uint
    x = np.array([1, 2, 3], dtype=np.int8)
    # np.prod(x).dtype == np.int
    a = np.array([1,2,3])
    np.cumprod(a) # intermediate results 1, 1*2
    a = np.array([[1, 2, 3], [4, 5, 6]])
    np.cumprod(a, dtype=float) # specify type of output
    np.cumprod(a, axis=0)
    np.cumprod(a,axis=1)
    np.ndim([[1,2,3],[4,5,6]])
    np.ndim(np.array([[1,2,3],[4,5,6]]))
    np.ndim(1)
    a = np.array([[1,2,3],[4,5,6]])
    np.size(a)
    np.size(a,1)
    np.size(a,0)
    np.around([0.37, 1.64])
    np.around([0.37, 1.64], decimals=1)
    np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
    np.around([1,2,3,11], decimals=1) # ndarray of ints is returned
    np.around([1,2,3,11], decimals=-1)
    a = np.array([[1, 2], [3, 4]])
    np.mean(a)
    np.mean(a, axis=0)
    np.mean(a, axis=1)
    a = np.zeros((2, 512*512), dtype=np.float32)
    a[0, :] = 1.0
    a[1, :] = 0.1
    np.mean(a)
    np.mean(a, dtype=np.float64)
    a = np.array([[1, 2], [3, 4]])
    np.std(a)
    np.std(a, axis=0)
    np.std(a, axis=1)
    a = np.zeros((2, 512*512), dtype=np.float32)
    a[0, :] = 1.0
    a[1, :] = 0.1
    np.std(a)
    np.std(a, dtype=np.float64)
    a = np.array([[1, 2], [3, 4]])
    np.var(a)
    np.var(a, axis=0)
    np.var(a, axis=1)
    a = np.zeros((2, 512*512), dtype=np.float32)
    a[0, :] = 1.0
    a[1, :] = 0.1
    np.var(a)
    np.var(a, dtype=np.float64)

def generate_default_blacklist():
    p = AutoBlacklistPolicy(gen_rule=True, append_rule=True)
    with p:
        test_ufunc()
        test_numeric()
        test_fromnumeric()
    p.save_rules()

if __name__ == '__main__':
    p = AutoBlacklistPolicy(gen_rule=True, append_rule=True)
    minpy.set_global_policy(p)
    logging.getLogger('minpy.dispatch.policy').setLevel(logging.DEBUG)
    test_ufunc()
    test_numeric()
    test_fromnumeric()
    logging.getLogger('minpy.dispatch.policy').setLevel(logging.WARN)
