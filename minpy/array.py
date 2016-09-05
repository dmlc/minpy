#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= unused-argument, protected-access,
# logging-format-interpolation
"""Base type for arrays."""
from __future__ import absolute_import
from __future__ import print_function

import itertools
import collections
import weakref

from minpy.array_variants import ArrayType
from minpy.array_variants import array_types
from minpy.array_variants import number_types
from minpy.context import current_context
from minpy.utils import log

import mxnet
import numpy

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)
# pylint: enable= invalid-name

GradRecord = collections.namedtuple('GradRecord',
                                    ['grad_func', 'result', 'primitive'])


class Node(object):
    """Node representing data with gradient information."""
    __slots__ = ['_value', '_partial_derivatives', '_partial_derivative_cache']

    def __init__(self, value):
        """Initialize."""
        self._value = weakref.ref(value)
        self._partial_derivatives = []
        self._partial_derivative_cache = {}

    def _get_value(self):
        r = self._value()
        if r is None:
            raise RuntimeError('Reference lost.')
        return r

    def add_partial_derivative(self, grad_func, res, prim):
        """Add partial derivative information.

        :param function grad_func: The function to calculate derivative with respect to result.
        :param Value res: Variable that represent the result of original function.
        :param Primitive prim: Primitive that the gradient function belongs to.
        """
        assert isinstance(res, Value), 'Result is not of type `Value`.'
        self._partial_derivatives.append(
            GradRecord(
                grad_func=grad_func, result=res, primitive=prim))

    def partial_derivative(self, target):
        """Get partial derivative. Mathematically, this function computes

        .. math::

           \\frac{\\partial target}{\\partial self}.

        :param Node target: Target variable to compute partial derivative.
        :return: Partial derivative.
        """

        def _call_partial_derivative(rec):
            """Helper function for calling gradient function.

            :param GradRecord rec: The gradient record to be called.
            :return: Gradient result.
            """
            # The gradient of the target with respect to the result node should already be
            # computed.
            result_grad = rec.result.node._partial_derivative_cache[target]
            result_grad_value = result_grad.get_data(rec.primitive._type)
            _logger.debug('Call derivative func of "{}".'.format(
                rec.primitive))
            # Call gradient function to compute input gradient from result gradient
            if rec.primitive.type == ArrayType.MXNET:
                with result_grad.context.as_mxnet_context() as ctx:
                    grad = rec.grad_func(result_grad_value)
            else:
                grad = rec.grad_func(result_grad_value)
            return grad

        # Array used to store intermediate gradients to be computed.
        pending_derivatives = []

        # Use DFS search to find all derivatives need to be computed in order to get the gradient
        # of final target.
        dfs_queue = [self]
        while len(dfs_queue) != 0:
            node = dfs_queue[-1]
            assert isinstance(target, Node), 'Type is not `Node`.'
            ready = True
            if target is not node:
                for rec in node._partial_derivatives:
                    n = rec.result.node
                    if target not in n._partial_derivative_cache:
                        dfs_queue.append(n)
                        ready = False
            # Successors all enqueued.
            if ready:
                dfs_queue.pop()
                if target not in node._partial_derivative_cache:
                    pending_derivatives.append(node)
                    # Init gradient buffer for accumulation.
                    node._partial_derivative_cache[target] = Value.wrap(
                        0.0 if isinstance(node._get_value(), Number) else
                        numpy.zeros(node._get_value().shape))

        # Compute gradient using chain rule.
        # The resolve order is the reversed order from target to input.
        for node in pending_derivatives:
            if node is target:
                # Current gradient node is the target node, the gradient is one.
                node._partial_derivative_cache[target] = Value.wrap(
                    1.0 if isinstance(node._get_value(), Number) else numpy.ones(
                        node._get_value().shape))
            else:
                # Call saved gradient function to compute gradient of each input.
                for rec in node._partial_derivatives:
                    node._partial_derivative_cache[target] += Value.wrap(
                        _call_partial_derivative(rec))

        return self._partial_derivative_cache[target]


class Value(object):
    # pylint: disable= no-self-use
    """Class for all possible values in MinPy.

    It contains the real underlying value and the gradient information for auto differentiation.
    It also defines common operators and redirects the call to the namespace dispatcher.
    """
    _ns = None

    def __init__(self, marked, context):
        self._marked_for_bp = marked
        if context is None:
            self._context = current_context()
        else:
            self._context = context

    @property
    def marked_for_bp(self):
        """Return whether the current Value will be used for autograd."""
        return self._marked_for_bp

    @property
    def context(self):
        return self._context

    @staticmethod
    def wrap(data, *args, **kwargs):
        """ Wrap given data into its corresponding wrapper class. For example, `numpy.ndarray`
        will be converted to `minpy.Array` while float number will become `minpy.Number`. The
        allowed array types are defined in `minpy.array_variants.array_types`; the allowed number
        types are defined in `minpy.array_variants.number_types`.
        """
        if data is None:
            return None
        dtype = type(data)
        if isinstance(data, Value):
            return data
        elif dtype in array_types.values():
            return Array(data, *args, **kwargs)
        elif dtype in itertools.chain(*number_types.values()):
            return Number(data, *args, **kwargs)
        else:
            raise TypeError('cannot wrap type: {}'.format(dtype))

    def __cmp__(self, other):
        raise NotImplementedError()

    def __eq__(self, other):
        return Value._ns.equal(self, other)

    def __ne__(self, other):
        return Value._ns.not_equal(self, other)

    def __lt__(self, other):
        return Value._ns.less(self, other)

    def __gt__(self, other):
        return Value._ns.greater(self, other)

    def __le__(self, other):
        return Value._ns.less_equal(self, other)

    def __ge__(self, other):
        return Value._ns.greater_equal(self, other)

    def __pos__(self):
        raise NotImplementedError()

    def __neg__(self):
        return Value._ns.negative(self)

    def __abs__(self):
        raise NotImplementedError()

    def __invert__(self):
        raise NotImplementedError()

    def __round__(self, nbits):
        raise NotImplementedError()

    def __floor__(self):
        raise NotImplementedError()

    def __ceil__(self):
        raise NotImplementedError()

    def __trunc__(self):
        raise NotImplementedError()

    def __add__(self, other):
        return Value._ns.add(self, other)

    def __sub__(self, other):
        return Value._ns.subtract(self, other)

    def __mul__(self, other):
        return Value._ns.multiply(self, other)

    def __floordiv__(self, other):
        raise NotImplementedError()

    def __div__(self, other):
        return Value._ns.divide(self, other)

    def __truediv__(self, other):
        return Value._ns.true_divide(self, other)

    def __mod__(self, other):
        return Value._ns.mod(self, other)

    def __divmod__(self, other):
        raise NotImplementedError()

    def __pow__(self, other):
        return Value._ns.power(self, other)

    def __lshift__(self, other):
        raise NotImplementedError()

    def __rshift__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        raise NotImplementedError()

    def __xor__(self, other):
        raise NotImplementedError()

    def __radd__(self, other):
        return Value._ns.add(other, self)

    def __rsub__(self, other):
        return Value._ns.subtract(other, self)

    def __rmul__(self, other):
        return Value._ns.multiply(other, self)

    def __rfloordiv__(self, other):
        raise NotImplementedError()

    def __rdiv__(self, other):
        return Value._ns.divide(other, self)

    def __rtruediv__(self, other):
        return Value._ns.true_divide(other, self)

    def __rmod__(self, other):
        return Value._ns.mod(other, self)

    def __rdivmod__(self, other):
        return Value._ns.mod(other, self)

    def __rpow__(self, other):
        return Value._ns.power(other, self)

    def __rlshift__(self, other):
        raise NotImplementedError()

    def __rrshift__(self, other):
        raise NotImplementedError()

    def __rand__(self, other):
        raise NotImplementedError()

    def __ror__(self, other):
        raise NotImplementedError()

    def __rxor__(self, other):
        raise NotImplementedError()

    def __iadd__(self, other):
        return Value._ns.add(self, other)

    def __isub__(self, other):
        return Value._ns.subtract(self, other)

    def __imul__(self, other):
        return Value._ns.multiply(self, other)

    def __ifloordiv__(self, other):
        raise NotImplementedError()

    def __idiv__(self, other):
        return Value._ns.divide(self, other)

    def __itruediv__(self, other):
        return Value._ns.true_divide(self, other)

    def __imod__(self, other):
        return Value._ns.mod(self, other)

    def __ipow__(self, other):
        return Value._ns.power(self, other)

    def __ilshift__(self, other):
        raise NotImplementedError()

    def __irshift__(self, other):
        raise NotImplementedError()

    def __iand__(self, other):
        raise NotImplementedError()

    def __ior__(self, other):
        raise NotImplementedError()

    def __ixor__(self, other):
        raise NotImplementedError()
    # pylint: enable= no-self-use


class Number(Value, float):
    """Class for numbers with derivative information"""
    __slots__ = ['_node', '_val', '_marked_for_bp']

    def __new__(cls, val, marked=False):
        return float.__new__(cls, val)

    def __init__(self, val, marked=False, context=None):
        super(Number, self).__init__(marked=marked, context=context)
        self._node = Node(self)
        self._val = val

    def __str__(self):
        return str(self._val)

    def __repr__(self):
        return self.__str__()

    def get_data(self, dtype):
        """Get data of given type. Directly return the underlying value here."""
        return self._val

    def asnumpy(self):
        """ Get data in numpy compatible type """
        return self._val

    @property
    def val(self):
        """ return the underlying value """
        return self._val

    @property
    def node(self):
        """ get node which contains derivative information from this array """
        return self._node


class Array(Value):
    """Base array type.

    It provides convenient methods for arithmetic operations. The Array class
    is used for:
    1. Redirect all special member functions to corresponding pure function.
    2. Redirect normal member functions to correct member functions of
    underlying array object.
    """
    __slots__ = ['_node', '_data', '_latest_version', '_marked_for_bp']
    __array_priority__ = 100.0  # highest priority when compute with numpy.ndarray

    def __init__(self, data, marked=False, context=None):
        super(Array, self).__init__(marked, context)
        self._data = {}
        self._node = Node(self)
        atype = Array.to_array_type(data)
        self._data[atype] = data
        self._latest_version = atype

    @staticmethod
    def to_array_type(arr):
        """ Return the type enum of the given array """
        atype = type(arr)
        if atype == array_types['numpy']:
            return ArrayType.NUMPY
        elif atype == array_types['mxnet']:
            return ArrayType.MXNET
        else:
            raise TypeError('Array data of type {} unknown.'.format(atype))

    def __str__(self):
        return str(self.get_data(ArrayType.NUMPY))

    def __repr__(self):
        return self.__str__()

    @property
    def node(self):
        """ get node which contains derivative information from this array """
        return self._node

    @property
    def context(self):
        return self._context

    @property
    def ndim(self):
        """ Number of array dimensions """
        # TODO (Yihe) add ndim in MXNet ndarray
        # return self._get_latest_data().ndim
        return self.get_data(ArrayType.NUMPY).ndim

    def has_type(self, atype):
        """ Return whether array data of given type exists in the underlying storage.
        """
        return atype in self._data.keys()

    def reshape(self, *args, **kwargs):
        """Function for reshape this array.

        Usage example:

        ::

            a = np.ones([10, 10])
            b = a.reshape([5, 20])
            b = a.reshape(5, 20)

        See `here <http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html>`_
        for further explanation.

        :param args: A single iterable or a sequence of integers representing a new shape.
            Although it being an iterable is not documented in official document, it is renowned and
            widely used in practice.
        :return: Reshaped array.
        """
        if len(args) == 1 and isinstance(args[0], collections.Iterable):
            new_shape = args[0]
        else:
            new_shape = tuple(x for x in args)
        if 'order' in kwargs and kwargs['order'] != 'C':
            raise NotImplementedError(
                'Orders other than C are not currently supported.')
        return Value._ns.reshape(self, new_shape)

    def dot(self, b, out=None):
        """ Function for dot production. """
        if out is not None:
            # TODO: Support out argument
            raise ValueError('out option is not supported.')
        return Value._ns.dot(self, b)

    def argmax(self, axis=None, out=None):
        """ Returns the indices of the maximum values along an axis

        :param axis: int. By default, the index is into the flattened array,
            otherwise along the specified axis.
        :param out: If provided, the result will be inserted into this array.
            It should be of the appropriate shape and dtype.
        :return: Array of indices into the array.
        """
        if out is not None:
            # TODO: Support out argument
            raise ValueError('out option is not supported.')
        return Value._ns.argmax(self, axis)

    def _synchronize_data(self):
        """ Synchronize the data of different array types. """
        if self._latest_version == ArrayType.MXNET:
            _logger.info('Copy from mxnet array to numpy array Node#{}'.format(
                id(self)))
            mxarray = self._data[ArrayType.MXNET]
            self._data[ArrayType.NUMPY] = mxarray.asnumpy()
        elif self._latest_version == ArrayType.NUMPY:
            _logger.info('Copy from numpy array to mxnet array Node#{}'.format(
                id(self)))
            nparray = self._data[ArrayType.NUMPY]
            self._data[ArrayType.MXNET] = mxnet.ndarray.array(
                nparray, ctx=self._context.as_mxnet_context())
        self._latest_version = None

    def enforce_data(self, dtype):
        """Enforce array data of given type."""
        if self._latest_version is not None and self._latest_version != dtype:
            self._synchronize_data()

    def get_data(self, dtype):
        """Get array data of given type."""
        self.enforce_data(dtype)
        return self._data[dtype]

    def _get_latest_data(self):
        """Return the latest version of the raw data"""
        if self._latest_version is not None:
            return self._data[self._latest_version]
        else:
            if self.has_type(ArrayType.NUMPY):
                return self._data[ArrayType.NUMPY]
            else:
                return self._data[ArrayType.MXNET]

    def asnumpy(self):
        """Get raw NumPy array.

        This will return a copied array of numpy.ndarray type
        """
        return numpy.array(self.get_data(ArrayType.NUMPY))

    def get_data_mutable(self, dtype):
        """Get exclusive access to array data of given type."""
        if self._latest_version is not None and self._latest_version != dtype:
            self._synchronize_data()
        self._latest_version = dtype
        return self._data[dtype]

    @property
    def shape(self):
        """ Get the shape of array """
        return self._get_latest_data().shape

    def __getitem__(self, index):
        """NumPy indexing operations.

        Currently `mxnet.ndarray` does not support full indexing, so there is an implicit
        conversion to NumPy array.
        """
        np_index = None
        to_np = lambda x: x if isinstance(x, slice) else Value.wrap(x).get_data(ArrayType.NUMPY)
        if isinstance(index, tuple):
            np_index = tuple(to_np(i) for i in index)
        else:
            np_index = to_np(index)
        return Value._ns._minpy_getitem(self, np_index)

    def __setitem__(self, index, val):
        """NumPy indexing operations.

        Currently `mxnet.ndarray` does not support full indexing, so there is an implicit
        conversion to NumPy array. Also note that this operation breaks gradient chain.
        """
        np_index = None
        np_val = Value.wrap(val).get_data(ArrayType.NUMPY)
        to_np = lambda x: x if isinstance(x, slice) else Value.wrap(x).get_data(ArrayType.NUMPY)
        if isinstance(index, tuple):
            np_index = tuple(to_np(i) for i in index)
        else:
            np_index = to_np(index)
        np_array = self.get_data_mutable(ArrayType.NUMPY)
        np_array.__setitem__(np_index, np_val)

    def __delitem__(self, index):
        """NumPy indexing operations.

        Currently `mxnet.ndarray` does not support full indexing, so there is an implicit
        conversion to NumPy array.  Also note that this operation breaks gradient chain.
        """
        self.get_data_mutable(ArrayType.NUMPY).__delitem(index)

    # pylint: disable= invalid-name
    @property
    def T(self):
        """ Get transposed array """
        return Value._ns.transpose(self)
    # pylint: enable= invalid-name

    @property
    def size(self):
        """ Get number of elements in the array """
        return self._get_latest_data().size
