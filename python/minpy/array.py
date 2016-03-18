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
from .array_variants import ArrayType
from .array_variants import array_types
from .array_variants import number_types
 
 #FIXME: should not import these; use array_invariants instead
import mxnet
import numpy

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
        if not target in self._partial_derivative_cache:
            if self is target:  # Partial derivative of self is one.
                self._partial_derivative_cache[target] = Value.wrap(1.0)
            else:
                def call(rec):
                    _logger.debug('Call derivative func of: {}'.format(rec[2]._func))
                    grad = rec[1].partial_derivative(target)
                    grad_value = grad.get_data(rec[2]._type)
                    return rec[0](grad_value)
                res = functools.reduce(
                        operator.add,
                        map(call, self._partial_derivatives),
                        0.0)
                self._partial_derivative_cache[target] = Value.wrap(res)
        return self._partial_derivative_cache[target]

class ArrayTypeMissingError(ValueError):
    pass

class UnknownArrayTypeError(ValueError):
    pass

class Value(object):
    _ns = None

    @staticmethod
    def wrap(d, *args, **kwargs):
        if d is None:
            return None
        t = type(d)
        if isinstance(d, Value):
            return d
        elif t in array_types.values():
            return Array(d, *args, **kwargs)
        elif t in itertools.chain(*number_types.values()):
            return Number(d, *args, **kwargs)
        else:
            raise UnknownArrayTypeError('cannot wrap type: {}'.format(t))

    def get_data(self, t):
        assert(False)
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
        return Value._ns.negative(self)

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
        return Value._ns.add(self, other)

    def __sub__(self, other):
        return Value._ns.subtract(self, other)

    def __mul__(self, other):
        return Value._ns.multiply(self, other)

    def __floordiv__(self, other):
        pass

    def __div__(self, other):
        return Value._ns.divide(self, other)

    def __truediv__(self, other):
        return Value._ns.true_divide(self, other)

    def __mod__(self, other):
        return Value._ns.mod(self, other)

    def __divmod__(self, other):
        pass

    def __pow__(self, other):
        return Value._ns.power(self, other)

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
        return Value._ns.add(other, self)

    def __rsub__(self, other):
        return Value._ns.subtract(other, self)

    def __rmul__(self, other):
        return Value._ns.multiply(other, self)

    def __rfloordiv__(self, other):
        pass

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
        return Value._ns.add(self, add)

    def __isub__(self, other):
        return Value._ns.subtract(self, other)

    def __imul__(self, other):
        return Value._ns.multiply(self, other)

    def __ifloordiv__(self, other):
        pass

    def __idiv__(self, other):
        return Value._ns.divide(self, other)

    def __itruediv__(self, other):
        return Value._ns.true_divide(self, other)

    def __imod__(self, other):
        return Value._ns.mod(self, other)

    def __ipow__(self, other):
        return Value._ns.power(self, other)

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

class Number(Value):
    """Class for numbers with derivative information"""
    __slots__ = ['_node', '_val', '_marked_for_bp']
    def __init__(self, val, marked=False):
        self._node = Node()
        self._val = val
        self._marked_for_bp = marked

    def __str__(self):
        return str(self._val)

    def get_data(self, t):
        """Get array data of given type."""
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
        if t == array_types['numpy']:
            return ArrayType.NUMPY
        elif t == array_types['mxnet']:
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

    def __getitem__(self, index):
        """ Get item only supports indexing type of numpy arrays right now,
        since mxnet.ndarray does not support int type & indexing
        """
        #XXX: Ideally, we should use the namespace dispatcher to select appropriate getitem.
        #     However, we don't have good support for int array now
        np_index = None
        if isinstance(index, tuple):
            np_index = tuple(Value.wrap(x).get_data(ArrayType.NUMPY) for x in index)
        else:
            np_index = Value.Wrap(index).get_data(ArrayType.NUMPY)
        return Value.wrap(self.get_data(ArrayType.NUMPY)[np_index])

    def __setitem__(self, index, val):
        """ Set item only supports indexing type of numpy arrays right now,
        since mxnet.ndarray does not support int type & indexing
        """
        #XXX: Ideally, we should use the namespace dispatcher to select appropriate setitem.
        #     However, we don't have good support for int array now
        np_index = None
        if isinstance(index, tuple):
            np_index = tuple(Value.wrap(x).get_data(ArrayType.NUMPY) for x in index)
        else:
            np_index = Value.Wrap(index).get_data(ArrayType.NUMPY)
        np_val = Value.wrap(val).get_data(ArrayType.NUMPY)
        self.get_data_mutable(ArrayType.NUMPY)[np_index] = np_val

class Primitive(object):
    """Primitive computation."""
    __slots__ = ['_func', '_grad_func', '_grad_func_kw', '_type', '_mutate_args', '_mutate_kw']

    def __init__(self, func, ty, mutate_args=[], mutate_kw=[]):
        """Initialize.
        Args:
            func: A function that performs the action.
        """
        self._func = func
        self._grad_func = {}
        self._grad_func_kw = {}
        self._type = ty
        self._mutate_args = mutate_args
        self._mutate_kw = mutate_kw

    @property
    def type(self):
        return self._type

    @property
    def typestr(self):
        if self._type == ArrayType.NUMPY:
            return "numpy"
        elif self._type == ArrayType.MXNET:
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
            An `Value` representing the result.

        Raises:
            IndexError:
                No corresponding gradient function.
            KeyError:
                No corresponding gradient function.
        """
        _logger.debug('Calling {} type {}'.format(self._func, self.typestr))

        def get_val(x, mutate):
            try:
                xv = Value.wrap(x)
                if mutate:
                    return xv.get_data_mutable(self._type)
                else:
                    return xv.get_data(self._type)
            except UnknownArrayTypeError: # if wrap failed, just return the original value
                pass
            return x
        # Get underlying data.
        arg_values = tuple(get_val(args[i], i in self._mutate_args) for i in range(len(args)))
        kwargs_values = {k: get_val(kwargs[k], k in self._mutate_kw) for k in kwargs}
        # Call the real function with raw value.
        result_value = self._func(*arg_values, **kwargs_values)
        # whether the result is on the bp path
        def scan(accum, x):
            if isinstance(x, Value):
                return operator.or_(accum, x._marked_for_bp)
            else:
                return accum
        # Check whether the result value is on the path of bp phase
        # If all the input arguments are not on the bp path, the result value is not as well.
        need_bp = functools.reduce(scan,
                         itertools.chain(args, kwargs.values()),
                         False)
        # Wrap the result raw value with wrapper and node.
        result = Value.wrap(result_value, marked=need_bp)
        if need_bp:
            # Record partial derivative paths, only for `Value` type values.
            # If no gradient function is defined, also omit it
            for i, arg in enumerate(args):
                if isinstance(arg, Value) and i < len(self._grad_func):
                    arg.node.add_partial_derivative(
                            self._grad_func[i](result_value, *arg_values, **kwargs_values),
                            result.node, self)
            for x in kwargs:
                if isinstance(kwargs[x], Value) and x in self._grad_func_kw:
                    kwargs[x].node.add_partial_derivative(
                            self._grad_func_kw[x](result_value, *arg_values, **kwargs_values),
                            result.node, self)
        return result

    def _enforce_input_type(self, f):
        def enforce(x):
            if self._type == ArrayType.NUMPY:
                x.enforce_data(ArrayType.NUMPY)
            elif self._type == ArrayType.MXNET:
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
        self._grad_func_kw[key] = func
        return self

    def def_grad_zero(self, argnum=0):
        self._grad_func[argnum] = lambda *args, **kwargs: lambda g: 0.0
