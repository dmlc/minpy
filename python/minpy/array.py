#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base type for arrays."""
from __future__ import absolute_import
from __future__ import print_function

import sys
import enum
import functools
import itertools
import operator
import logging

from .utils import log
from .utils import common
#import typing
from .array_variants import FunctionType
from .array_variants import ArrayType
from .array_variants import allowed_types

import mxnet #FIXME: should not import this; use array_invariants instead

_logger = log.get_logger(__name__)

class Node(object):
    """Node representing data with gradient information."""
    __slots__ = ['_partial_derivatives', '_partial_derivative_cache']

    def __init__(self):
        """Initialize."""
        self._partial_derivatives = []
        self._partial_derivative_cache = {}

    def add_partial_derivative(self, grad_func, res, prim):
        """ Add partial derivative information

        :param function grad_func: the function to calculate derivative with respect to res
        :param Node res: variable that represent the target of derivative
        :param Primitive prim: the primitive that the gradient function belongs to
        """
        _logger.debug('Adding partial derivative to Node #{}'.format(id(self)))
        assert(isinstance(res, Node))
        self._partial_derivatives.append((grad_func, res, prim))

    def partial_derivative(self, target):
        """ Add partial derivative information

        :param Node target: target variable to compute partial derivative
        """
        assert(isinstance(target, Node))
        if target in self._partial_derivative_cache:
            return self._partial_derivative_cache[target]
        else:
            if self is target:  # Partial derivative of self is one.
                return 1.0
            else:
                def call(rec):
                    _logger.debug('Call derivative func of: {}'.format(rec[2]._func))
                    return rec[0](rec[1].partial_derivative(target))
                res = functools.reduce(
                        operator.add,
                        map(call, self._partial_derivatives),
                        0.0)
                self._partial_derivative_cache[target] = res
                return res

class ArrayTypeMissingError(ValueError):
    pass

class UnknownArrayTypeError(ValueError):
    pass

class Value(object):
    pass

class Array(Value):
    """Base array type.

    It provides convenient methods for arithmetic operations. The Array class
    is used for:
    1. Redirect all special member functions to corresponding pure function.
    2. Redirect normal member functions to correct member functions of
    underlying array object.
    """
    __slots__ = ['_node', '_data', '_latest_version', '_marked_for_bp']

    _ns = None

    def __init__(self, data, marked=False):
        self._data = {}
        self._node = Node()
        t = Array.to_array_type(data)
        self._data[t] = data
        self._latest_version = t
        self._marked_for_bp = marked

    @staticmethod
    def to_array_type(arr):
        t = type(arr)
        if t in allowed_types['numpy']:
            return ArrayType.NUMPY
        elif t in allowed_types['mxnet']:
            return ArrayType.MXNET
        else:
            raise UnknownArrayTypeError(
                'Array data of type {} unknown.'.format(t))

    def __str__(self):
        return str(self.get_data(ArrayType.NUMPY))

    @property
    def node(self):
        """ get node which contains derivative information from this array """
        return self._node

    def has_type(self, t):
        """Return whether array data of given type exists in the underlying storage.
        """
        return t in self._data.keys()

    def _synchronize_data(self):
        if self._latest_version == ArrayType.MXNET:
            _logger.info('Copy from mxnet array to numpy array Node#{}'.format(id(self)))
            mxarray = self._data[ArrayType.MXNET]
            self._data[ArrayType.NUMPY] = mxarray.asnumpy()
        elif self._latest_version == ArrayType.NUMPY:
            _logger.info('Copy from numpy array to mxnet array Node#{}'.format(id(self)))
            nparray = self._data[ArrayType.NUMPY]
            self._data[ArrayType.MXNET] = mxnet.ndarray.array(nparray, ctx=mxnet.gpu(0)) # TODO on which device ?
        self._latest_version = None

    def enforce_data(self, t):
        """Enforce array data of given type."""
        if self._latest_version is not None and self._latest_version != t:
            self._synchronize_data()
            self._latest_version = None

    def get_data(self, t):
        """Get array data of given type."""
        self.enforce_data(t)
        return self._data[t]

    def asnumpy(self):
        """Get raw NumPy array.

        This will return a copied array of numpy.ndarray type
        """
        return numpy.array(self.get_data(ArrayType.NUMPY))

    def get_data_mutable(self, t):
        """Get exclusive access to array data of given type."""
        if self._latest_version is not None and self._latest_version != t:
            self._synchronize_data()
        self._latest_version = t
        return self._data[t]

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
        return Array._ns.negative(self)

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
        return Array._ns.add(self, other)

    def __sub__(self, other):
        return Array._ns.subtract(self, other)

    def __mul__(self, other):
        return Array._ns.multiply(self, other)

    def __floordiv__(self, other):
        pass

    def __div__(self, other):
        return Array._ns.divide(self, other)

    def __truediv__(self, other):
        return Array._ns.true_divide(self, other)

    def __mod__(self, other):
        return Array._ns.mod(self, other)

    def __divmod__(self, other):
        pass

    def __pow__(self, other):
        return Array._ns.power(self, other)

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
        return Array._ns.add(other, self)

    def __rsub__(self, other):
        return Array._ns.subtract(other, self)

    def __rmul__(self, other):
        return Array._ns.multiply(other, self)

    def __rfloordiv__(self, other):
        pass

    def __rdiv__(self, other):
        return Array._ns.divide(other, self)

    def __rtruediv__(self, other):
        return Array._ns.true_divide(other, self)

    def __rmod__(self, other):
        return Array._ns.mod(other, self)

    def __rdivmod__(self, other):
        return Array._ns.mod(other, self)

    def __rpow__(self, other):
        return Array._ns.power(other, self)

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
        return Array._ns.add(other, self)

    def __isub__(self, other):
        return Array._ns.subtract(other, self)

    def __imul__(self, other):
        return Array._ns.multiply(other, self)

    def __ifloordiv__(self, other):
        pass

    def __idiv__(self, other):
        return Array._ns.divide(other, self)

    def __itruediv__(self, other):
        return Array._ns.true_divide(other, self)

    def __imod__(self, other):
        return Array._ns.mod(other, self)

    def __ipow__(self, other):
        return Array._ns.power(other, self)

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

class Primitive(object):
    """Primitive computation."""
    __slots__ = ['_func', '_grad_func', '_grad_func_kw', '_type']

    def __init__(self, func, ty):
        """Initialize.
        Args:
            func: A function that performs the action.
        """
        self._func = func
        self._grad_func = {}
        self._grad_func_kw = {}
        self._type = ty

    @property
    def type(self):
        return self._type

    @property
    def typestr(self):
        if self._type == FunctionType.NUMPY:
            return "numpy"
        elif self._type == FunctionType.MXNET:
            return "mxnet"
        else:
            return "N/A"

    def __str__(self):
        return self._func.__name__

    def __call__(self, *args, **kwargs):
        """Call wrapped function.
        Args:
            *args:
                Arguments for the wrapped function.
            **kwargs:
                Arguments for the wrapped function.

        Returns:
            An `Array` representing the result.

        Raises:
            IndexError:
                No corresponding gradient function.
            KeyError:
                No corresponding gradient function.
        """
        _logger.debug('Calling {} type {}'.format(self._func, self.typestr))

        def get_val(x):
            return x.get_data(self._type) if isinstance(x, Array) else x
        # Get underlying data.
        arg_values = tuple(map(get_val, args))
        kwargs_values = {x: get_val(kwargs[x]) for x in kwargs}
        # Call the real function with raw value.
        result_value = self._func(*arg_values, **kwargs_values)
        # whether the result is on the bp path
        def scan(accum, x):
            if isinstance(x, Array):
                return operator.or_(accum, x._marked_for_bp)
            else:
                return accum
        # Check whether the result value is on the path of bp phase
        # If all the input arguments are not on the bp path, the result value is not as well.
        need_bp = functools.reduce(scan,
                         itertools.chain(args, kwargs.values()),
                         False)
        result_value_type = type(result_value)
        if need_bp:
            # Wrap the result raw value with wrapper and node.
            result = Array(result_value, marked=True)
            # Record partial derivative paths, only for `Array` type values.
            # If no gradient function is defined, also omit it
            for i, arg in enumerate(args):
                if isinstance(arg, Array) and i < len(self._grad_func):
                    arg.node.add_partial_derivative(
                            self._grad_func[i](result_value, *arg_values, **kwargs_values),
                            result.node, self)
            for x in kwargs:
                if isinstance(arg, Array) and x in self._grad_func_kw:
                    arg.node.add_partial_derivative(
                            self._grad_func_kw[x](result_value, *arg_values, **kwargs_values),
                            result.node, self)
        else:
            result = result_value
        return result

    def _enforce_input_type(self, f):
        def enforce(x):
            if self._type == FunctionType.NUMPY:
                x.enforce_data(ArrayType.NUMPY)
            elif self._type == FunctionType.MXNET:
                x.enforce_data(ArrayType.MXNET)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*tuple(map(enforce, args)), **{x: enforce(kwargs[x]) for x in kwargs})
        return wrapper

    def def_grad(self, func, argnum=0):
        """Define gradient function.
        Args:
            func:
                Gradient function.
            argnum:
                Index of the argument.

        Return:
            self instance for multiple def_grad in one statement
        """
        #self._grad_func[argnum] = self._enforce_input_type(func)
        self._grad_func[argnum] = func
        return self

    def def_grad_kw(self, func, key):
        """Define gradient function.
        Args:
            func:
                Gradient function.
            key:
                Key name of the argument.
        """
        #self._grad_func[key] = self._enforce_input_type(func)
        self._grad_func[key] = func
        return self

    def def_grad_zero(self, argnum=0):
        self._grad_func[argnum] = lambda *args, **kwargs: lambda g: 0.0
