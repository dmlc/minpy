#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core gradient calculation."""
import functools
import operator
from ..utils import log
from .. import array

_logger = log.get_logger(__name__, log.logging.WARNING)


class Node(object):
    """Node representing data with gradient information."""
    __slots__ = ['_val', '_partial_derivatives', '_partial_derivative_cache']

    # TODO better use weakref here, but let's trust Python's GC for now
    _val = None
    _partial_derivatives = []
    _partial_derivative_cache = []

    def __init__(self, val: array.Array):
        """Initialize.

        :param array.Array val: Value to wrap.
        """
        self._val = val

    def __str__(self):
        """Get string representation.

        Return:
            A string representation.
        """
        return 'Node({})'.format(self._val)

    @property
    def val(self) -> array.Array:
        return self._val

    def add_partial_derivative(self, func, res):
        _logger.info('Adding partial derivative to {}: {}'.format(id(self),
                                                                  self))
        self._partial_derivatives.append((func, res))

    def partial_derivative(self, target):
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
                _logger.info('Partial derivative id: {}, shape: {}, value: {}'.
                             format(id(self), self.val.shape, res))
                return res


# TODO move next to numpy_primitive and mxnet_primitive, probably need
# different implementation
"""
class Primitive(object):
    """Primitive computation."""
    __slots__ = ['_func', '_grad_func', '_grad_func_kw', '_type']

    def __init__(self, func):
        """Initialize.

    Args:
        func:
            A function that performs the action.
    """
        self._func = func
        self._grad_func = {}
        self._grad_func_kw = {}
        self._type = None  # will be set later by registry

    def __call__(self, *args, **kwargs):
        """Call wrapped function.

    Args:
        *args:
            Arguments for the wrapped function.
        **kwargs:
            Arguments for the wrapped function.

    Returns:
        A `Node` representing the result.

    Raises:
        IndexError:
            No corresponding gradient function.
        KeyError:
            No corresponding gradient function.
    """
        _logger.info('Calling {}'.format(self._func))

        def get_val(x):
            return x._val if isinstance(x, Node) else x
        # Get underlying data.
        arg_values = tuple(map(get_val, args))
        kwargs_values = {x: get_val(kwargs[x]) for x in kwargs}
        # Call the real function with raw value.
        result_value = self._func(*arg_values, **kwargs_values)
        # Wrap the result raw value with wrapper and node.
        result = Node(result_value)
        # Record partial derivative paths, only for `Node` type values.
        for i, arg in enumerate(args):
            if isinstance(arg, Node):
                arg.add_partial_derivative(self._grad_func[i](
                    result_value, *arg_values, **kwargs_values), result)
        for x in kwargs:
            if isinstance(arg, Node):
                arg.add_partial_derivative(self._grad_func_kw[x](
                    result_value, *arg_values, **kwargs_values), result)
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
"""


def grad(func, argnum=0):
    @functools.wraps(func)
    def wrapped(*args):
        nodes = tuple(map(Node, args))
        result_node = func(*nodes)
        return nodes[argnum].partial_derivative(result_node)
    return wrapped
