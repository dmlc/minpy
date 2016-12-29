#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=logging-format-interpolation
"""Primitive class definitions."""
from __future__ import absolute_import
from __future__ import print_function

import functools
import collections

from . import array
from .array_variants import ArrayType
from .array_variants import variants
from .array_variants import variants_repr
from . import context
from .utils import log
from . import tape

_logger = log.get_logger(__name__)  # pylint: disable= invalid-name
GradFunc = collections.namedtuple('GradFunc', ['f', 'multi_grad_indices'])


class NoGradientFuncError(ValueError):
    """Error of no required gradient func defined for back propagation.

    Parameters
    ----------
    func_name : str
        Name of the primitive without gradient definition.
    arg : str or int
        If str, arg is the name of a keyword argument. If int, arg is the
        position of a positional argument.
    """

    def __init__(self, func_name, arg):
        if isinstance(arg, str):
            arg_name = 'keyword argument'
        else:
            arg_name = 'argument'
        msg = "No partial derivative of {} defined on {} {}.".format(func_name, arg_name, arg)
        super(NoGradientFuncError, self).__init__(msg)


class FakeGradFunc(object):
    """Fake gradient func for raising NoGradientFuncError at proper place.

    Parameters
    ----------
    func_name : str
        Name of the primitive without gradient definition.
    arg : str or int
        If str, arg is the name of a keyword argument. If int, arg is the
        position of a positional argument.
    """

    def __init__(self, func_name, arg):
        self.func_name = func_name
        self.arg = arg

    def __call__(self, grad_value):
        raise NoGradientFuncError(self.func_name, self.arg)


class Primitive(object):
    """Class for primitives. It includes both forward function and gradient definition."""
    # pylint: disable=too-many-arguments
    __slots__ = [
        '_func',
        '_grad_func',
        '_grad_func_kw',
        '_type',
        '_mutate_args',
        '_mutate_kw',
        '_type_str',
        '__name__',
    ]

    def __init__(self, func, ty, mutate_args=None, mutate_kw=None, type_str=None):
        """Initialize.

        Parameters
        ----------
        func
            A function that performs the forward computation.
        ty
            Type of primitive.
        mutate_args
            The indices of arguments that need to be mutated.
        mutate_kw
            The keywords of arguments that need to be mutated.
        type_str : None or str
            Specify type string for debugging. If not specified, infer by `ty`.
        """
        self._func = func
        self._grad_func = {}
        self._grad_func_kw = {}
        self._type = ty
        self._mutate_args = [] if mutate_args is None else mutate_args
        self._mutate_kw = [] if mutate_kw is None else mutate_kw
        if type_str is None:
            self._type_str = variants_repr[ty] + ' op'
        else:
            self._type_str = type_str
        self.__name__ = func.__name__

    @property
    def type(self):
        """Return the type of the primitive.

        Either :attribute:`ArrayType.NUMPY` or :attribute:`ArrayType.MXNET`.
        """
        return self._type

    @property
    def typestr(self):
        """Return the string representation of primitive type.

        Returns
        -------
        string
            String representation.
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
        """Wrap args for `call` method."""
        return self.call(args, kwargs)

    def call(self, args, kwargs):  # pylint: disable= too-many-branches
        """Call wrapped function -- compact argument ver.

        Parameters
        ----------
        args
            Arguments for the wrapped function.
        kwargs
            Arguments for the wrapped function.

        Returns
        -------
        array.Value
            An :class:`array.Value` representing the result.

        Raises
        ------
        IndexError, KeyError
            No corresponding gradient function.
        """
        # pylint: disable=too-many-locals, too-many-nested-blocks
        _logger.debug('Calling "{}" type "{}".'.format(self._func, self.typestr))

        # Check whether the result value is on the path of bp phase.
        # If all the input arguments are not on the bp path, the result value
        # is not as well.
        need_bp = [False]
        current_tape = tape.global_tape()

        def get_val(val, mutate):
            """Get value of array."""
            wrapped_val = array.wrap(val)
            if hasattr(wrapped_val, '_minpy_value_id'):
                # XXX: Use list of only one element to avoid variable assignment. # pylint: disable= fixme
                if not need_bp[0] and wrapped_val.is_marked_for_bp(current_tape):
                    need_bp[0] = True
                if mutate:
                    return wrapped_val.get_data_mutable(self._type)
                else:
                    return wrapped_val.get_data(self._type)
            return wrapped_val

        # Get underlying data.
        arg_values = tuple(
            get_val(arg, i in self._mutate_args) for i, arg in enumerate(args))
        kwargs_values = {
            k: get_val(arg, k in self._mutate_kw)
            for k, arg in kwargs.items()
        }
        # Call the real function with raw value.
        if self.type == ArrayType.MXNET:
            with context.current_context().as_mxnet_context():
                result_value = self._func(*arg_values, **kwargs_values)
        else:
            result_value = self._func(*arg_values, **kwargs_values)
        # If you want to do profiling, try to use `minprof(func)`.
        # result_value = minprof(self._func)(*arg_values, **kwargs_values)
        # Wrap the result raw value with wrapper and node.
        if isinstance(result_value, tuple):
            # Multiple return values.
            result = tuple(array.wrap(ret) for ret in result_value)
        else:
            result = array.wrap(result_value)
        if need_bp[0]:
            if isinstance(result, tuple):
                for res in result:
                    res.mark_for_bp(current_tape)
            else:
                result.mark_for_bp(current_tape)  # pylint: disable= no-member
            # Record partial derivative paths, only for `array.Value` type values.
            # If no gradient function is defined, also omit it
            visited_arg_indices = set()

            def get_context(result):
                """Get context of result."""
                if isinstance(result, array.Value):
                    return result.context
                else:
                    return get_context(result[0])

            def context_wrapper(func):
                """A context wrapper only for gradient function."""

                @functools.wraps(func)
                def wrapped(result):  # pylint: disable= missing-docstring
                    with get_context(result).as_mxnet_context():
                        return func(result)

                return wrapped

            def raw_value_wrapper(func):
                """Unwrap Value for gradient function."""

                @functools.wraps(func)
                def wrapped(result):  # pylint: disable= missing-docstring
                    if isinstance(result, (tuple, list)):
                        result = (elm.get_data(self.type) for elm in result)
                    else:
                        result = result.get_data(self.type)  # pylint: disable= no-member
                    return func(result)

                return wrapped

            for i, arg in enumerate(args):
                if i in visited_arg_indices:
                    continue
                else:
                    visited_arg_indices.add(i)
                if isinstance(arg, array.Value) and arg.is_marked_for_bp(current_tape):
                    if i not in self._grad_func:
                        _logger.debug('Partial derivative of {} "{}" on'
                                      'argument {} is not defined.'
                                      .format(self._type_str, self._func.__name__, i))
                        grad_func = FakeGradFunc(self._type_str + ' ' + self._func.__name__, i)
                        owner = arg
                    else:
                        _logger.debug(
                            'Adding partial derivative to func "{}" on argument {}.'.
                            format(self._func, i))
                        grad_func_rec = self._grad_func[i]
                        # Save forward results and arguments in the gradient function closure
                        # for later use.
                        grad_func = grad_func_rec.f(result_value, *arg_values, **kwargs_values)

                        if grad_func_rec.multi_grad_indices is None:
                            # Derivative function of each argument is defined separately.
                            owner = arg
                        else:
                            # Derivative function could compute gradients of multiple arguments
                            # in one call.
                            owner = []
                            for grad_index in grad_func_rec.multi_grad_indices:
                                if (isinstance(args[grad_index], array.Value) and \
                                    args[grad_index].is_marked_for_bp(current_tape)):
                                    owner.append(args[grad_index])
                                else:
                                    # Use None as placeholder for arguments that do not require
                                    # gradient computation.
                                    owner.append(None)
                                visited_arg_indices.add(grad_index)
                        grad_func = raw_value_wrapper(grad_func)  # pylint: disable=redefined-variable-type
                        if self.type == ArrayType.MXNET:
                            grad_func = context_wrapper(grad_func)
                    current_tape.add_partial_derivative(grad_func, owner,
                                                        result)
            for k, arg in kwargs.items():
                if isinstance(arg, array.Value) and arg.is_marked_for_bp(
                        current_tape):
                    if k not in self._grad_func_kw:
                        _logger.debug(
                            'Partial derivative of {} "{}" on keyword argument "{}"'
                            'is not defined.'.format(self._type_str, self._func.__name__, k))
                        grad_func = FakeGradFunc(self._type_str + ' ' + self._func.__name__, k)
                    else:
                        _logger.debug(
                            'Adding partial derivative to func "{}" on keyword argument "{}".'.
                            format(self._func, k))
                        grad_func_rec = self._grad_func_kw[k]
                        grad_func = grad_func_rec.f(result_value, *arg_values,
                                                    **kwargs_values)
                        grad_func = raw_value_wrapper(grad_func)
                        if self.type == ArrayType.MXNET:
                            grad_func = context_wrapper(grad_func)
                    current_tape.add_partial_derivative(grad_func, arg, result)
        return result
        # pylint: enable=invalid-name, too-many-locals

    def def_grad(self, func, argnum=0):
        """Define gradient function.

        Parameters
        ----------
        func
            Gradient function, in the form of
            lambda ans, *args, **kwargs: lambda g: real_grad_func
        argnum
            Index of the argument.
        """
        self._grad_func[argnum] = GradFunc(f=func, multi_grad_indices=None)
        return self

    def def_grad_kw(self, func, key):
        """Define gradient function.

        Parameters
        ----------
        func
            Gradient function.
        key
            Key name of the argument.
        """
        self._grad_func_kw[key] = GradFunc(f=func, multi_grad_indices=None)
        return self

    def def_grad_zero(self, argnum=0):
        """Define zero gradient

        Parameters
        ----------
        argnum
            Index of the argument.

        Returns
        -------
        Primitive
            Self.
        """
        self.def_grad(lambda *args, **kwargs: lambda g: 0.0, argnum)
        return self

    def def_multiple_grad(self, func, argnums):
        """Define multiple gradients with one function.

        Parameters
        ----------
        func
            Gradient function.
        argnums
            Indices of the arguments.
        """
        assert isinstance(argnums, tuple), 'Indexes must be of type tuple.'
        assert len(argnums) == len(set(argnums)), 'Duplicate entries.'

        for argnum in argnums:
            assert isinstance(argnum,
                              int), 'Indexes must be tuple of integers.'
            self._grad_func[argnum] = GradFunc(
                f=func, multi_grad_indices=argnums)
        return self

    def gradable(self, bp_args, bp_kwargs):
        """Return whether the primitive has gradient function defined.

        Parameters
        ----------
        bp_args
            Positional arguments that need back propagation.
        bp_kwargs
            Keyword arguments that need back propagation.

        Returns
        -------
        bool
            Whether all the arguments have gradients defined.
        """
        return all(i in self._grad_func for i in bp_args) and \
               all(i in self._grad_func_kw for i in bp_kwargs)


def customop(op_type):
    """Wrapper for CustomOp.

    Parameters
    ----------
    type : str
        Type of the custom op. Input vars are converted to the given type. 'minpy' or 'mxnet'.

    Returns
    -------
    Returns the proper wrapper by given type.
    """

    def customop_wrapper(func):
        """Real wrapper"""
        if op_type == 'numpy' or op_type == 'mxnet':
            return Primitive(func, variants[op_type], type_str='customized op')
        else:
            raise ValueError('Wrong type for CustomOp. Must be either "numpy" or "mxnet"')

    return customop_wrapper
