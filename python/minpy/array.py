#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base type for arrays."""
from __future__ import absolute_import

from .utils import log
from .utils import common
#import typing
from .array_variants import FunctionType
from .array_variants import ArrayType
import mxnet
import numpy
import sys

_logger = log.get_logger(__name__)

class Node(object):
    """Node representing data with gradient information."""
    __slots__ = ['_partial_derivatives', '_partial_derivative_cache']

    _partial_derivatives = []
    _partial_derivative_cache = []

    def __init__(self):
        """Initialize."""
        pass

    def __str__(self):
        """Get string representation.

        Return:
            A string representation.
        """
        return 'Node({})'.format(self)

    def add_partial_derivative(self, func, res):
        """ Add partial derivative information

        :param function func: the function to calculate derivative with respect to res
        :param Node res: variable that represent the target of derivative
        """
        _logger.info('Adding partial derivative to {}: {}'.format(id(self),
                                                                  self))
        assert(isinstance(res, Node))
        self._partial_derivatives.append((func, res))

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
                res = functools.reduce(operator.add, map(
                    lambda x: x[0](x[1].partial_derivative(target)),
                    self._partial_derivatives), 0.0)
                self._partial_derivative_cache[target] = res
                #_logger.info('Partial derivative id: {}, shape: {}, value: {}'.
                             #format(id(self), self.val.shape, res))
                return res

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
    __slots__ = ['_node', '_data']

    _node = Node()
    _data = {}

    @staticmethod
    #def to_array_type(arr: typing.Union[numpy.ndarray, mxnet.narray.NArray]
                      #) -> ArrayType:
    def to_array_type(arr):
        t = type(arr)
        if t == numpy.ndarray:
            return ArrayType.NUMPY
        elif t == mxnet.nd.NArray:
            return ArrayType.MXNET
        else:
            raise UnknownArrayTypeError(
                'Array data of type {} unknown.'.format(t))

    @staticmethod
    #def to_real_type(arr: ArrayType) -> type:
    def to_real_type(arr):
        if arr == ArrayType.NUMPY:
            return numpy.ndarray
        elif arr == ArrayType.MXNET:
            return mxnet.nd.NArray
        else:
            raise UnknownArrayTypeError(
                'Array data of type {} unknown.'.format(arr))

    @property
    def node(self):
        """ get node which contains derivative information from this array """
        return self._node

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

    #def create_data(self, t: ArrayType):
    def create_data(self, t):
        """Create data of given type."""
        if t not in self._data:
            if t == ArrayType.NUMPY:
                mxarray = self.get_data(ArrayType.MXNET)
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
        return self._type;

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
        _logger.info('Calling {}'.format(self._func))

        def get_val(x):
            return x.get_data(self._type) if isinstance(x, Array) else x
        # Get underlying data.
        arg_values = tuple(map(get_val, args))
        kwargs_values = {x: get_val(kwargs[x]) for x in kwargs}
        # Call the real function with raw value.
        result_value = self._func(*arg_values, **kwargs_values)
        # Wrap the result raw value with wrapper and node.
        result = Array(result_value)
        # Record partial derivative paths, only for `Array` type values.
        for i, arg in enumerate(args):
            if isinstance(arg, Array):
                arg.node.add_partial_derivative(
                        self._grad_func[i](result_value, *arg_values, **kwargs_values),
                        result.node)
        for x in kwargs:
            if isinstance(arg, Array):
                arg.node.add_partial_derivative(
                        self._grad_func_kw[x](result_value, *arg_values, **kwargs_values),
                        result.node)
        return result

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
        self._grad_func[key] = func

    def def_grad_zero(self, argnum=0):
        self._grad_func[argnum] = lambda *args, **kwargs: lambda g: 0.0
