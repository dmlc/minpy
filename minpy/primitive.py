#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Primitive class definitions."""
from __future__ import absolute_import
from __future__ import print_function

import functools
import itertools
import operator

from minpy.array import Value
from minpy.array_variants import ArrayType
from minpy.array_variants import array_types
from minpy.array_variants import number_types
from minpy.context import Context, current_context
from minpy.utils import log

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)
# pylint: enable= invalid-name

class Primitive(object):
    """Class for primitives. It includes both forward function and gradient definition."""
    __slots__ = [
        '_func',
        '_grad_func',
        '_grad_func_kw',
        '_type',
        '_mutate_args',
        '_mutate_kw',
    ]

    def __init__(self, func, ty, mutate_args=None, mutate_kw=None):
        """Initialize.
        Args:
            func: A function that performs the action.
        """
        self._func = func
        self._grad_func = {}
        self._grad_func_kw = {}
        self._type = ty
        self._mutate_args = [] if mutate_args is None else mutate_args
        self._mutate_kw = [] if mutate_kw is None else mutate_kw

    @property
    def type(self):
        """ Return the type of the primitive (ArrayType.NUMPY or ArrayType.MXNET) """
        return self._type

    @property
    def typestr(self):
        """Return the string representation of primitive type.

        :return: String representation.
        """
        if self._type == ArrayType.NUMPY:
            return "NumPy"
        elif self._type == ArrayType.MXNET:
            return "MXNet"
        else:
            raise NotImplementedError()

    def __str__(self):
        return self._func.__name__

    def __call__(self, *args, **kwargs):
        """Call wrapped function.

        :param args: Arguments for the wrapped function.
        :param kwargs: Arguments for the wrapped function.
        :return: An :class:`array.Value` representing the result.
        :raises IndexError: No corresponding gradient function.
        :raises KeyError: No corresponding gradient function.
        """
        # pylint: disable= missing-docstring, invalid-name
        _logger.debug('Calling {} type {}.'.format(self._func, self.typestr))

        def get_val(x, mutate):
            try:
                xv = Value.wrap(x)
                if mutate:
                    return xv.get_data_mutable(self._type)
                else:
                    return xv.get_data(self._type)
            # If wrap failed, just return the original value.
            except TypeError:
                return x
        # Get underlying data.
        arg_values = tuple(
            get_val(args[i], i in self._mutate_args) for i in range(len(args)))
        kwargs_values = {
            k: get_val(kwargs[k], k in self._mutate_kw)
            for k in kwargs
        }
        # Call the real function with raw value.
        if self.type == ArrayType.MXNET:
            with current_context().as_mxnet_context() as ctx:
                result_value = self._func(*arg_values, **kwargs_values)
        else:
            result_value = self._func(*arg_values, **kwargs_values)
        # if you want to do profiling, try to use minprof(<func>):
        # result_value = minprof(self._func)(*arg_values, **kwargs_values)

        # whether the result is on the bp path
        def scan(accum, x):
            if isinstance(x, Value):
                return operator.or_(accum, x._marked_for_bp)
            else:
                return accum
        # Check whether the result value is on the path of bp phase
        # If all the input arguments are not on the bp path, the result value
        # is not as well.
        need_bp = functools.reduce(scan, itertools.chain(
            args, kwargs.values()), False)
        # Wrap the result raw value with wrapper and node.
        result = Value.wrap(result_value, marked=need_bp)
        if need_bp:
            # Record partial derivative paths, only for `Value` type values.
            # If no gradient function is defined, also omit it
            for i, arg in enumerate(args):
                if isinstance(arg, Value) and arg.marked_for_bp:
                    if i not in self._grad_func:
                        _logger.warn('Partial derivative of func "{}" on #{} \
                            arg is not defined.'
                                     .format(self._func.__name__, i))
                        continue
                    _logger.debug(
                        'Adding partial derivative to func "{}" on #{} arg.'.format(
                            self._func, i))
                    arg.node.add_partial_derivative(self._grad_func[i](
                        result_value, *arg_values, **kwargs_values),
                                                    result.node, self)
            for k, arg in kwargs.items():
                if isinstance(arg, Value) and arg.marked_for_bp:
                    if k not in self._grad_func_kw:
                        _logger.warn(
                            'Partial derivative of func "{}" on kwarg "{}"\
                            is not defined.'.format(self._func.__name__, k))
                        continue
                    _logger.debug(
                        'Adding partial derivative to func "{}" on kwarg "{}".'.format(
                            self._func, k))
                    arg.node.add_partial_derivative(self._grad_func_kw[k](
                        result_value, *arg_values, **kwargs_values),
                                                    result.node, self)
        return result
        # pylint: enable= missing-docstring, invalid-name

    def def_grad(self, func, argnum=0):
        """Define gradient function.

        :param func: Gradient function.
        :param argnum: Index of the argument.
        :return: Self.
        """
        self._grad_func[argnum] = func
        return self

    def def_grad_kw(self, func, key):
        """Define gradient function.

        :param func: Gradient function.
        :param key: Key name of the argument.
        :return: Self.
        """
        self._grad_func_kw[key] = func
        return self

    def def_grad_zero(self, argnum=0):
        """Define zero gradient

        :param argnum: Index of the argument.
        :return: Self.
        """
        self._grad_func[argnum] = lambda *args, **kwargs: lambda g: 0.0
        return self

    def gradable(self, bp_args, bp_kwargs):
        """Return whether the primitive has gradient function defined.

        :param tuple bp_args: Positional arguments that need back propagation.
        :param tuple bp_kwargs: Keyword arguments that need back propagation.
        :return: Whether all the arguments have gradients defined.
        """
        for i in bp_args:
            if i not in self._grad_func:
                return False
        for i in bp_kwargs:
            if i not in self._grad_func_kw:
                return False
        return True
