#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= protected-access, invalid-name
"""Core gradient calculation."""
from __future__ import absolute_import
from __future__ import print_function

import mxnet as mx
import functools

from minpy.array import Value
from minpy.array_variants import ArrayType, array_types, number_types
from minpy.context import current_context
from minpy.primitive import Primitive
from minpy.utils import log

_logger = log.get_logger(__name__)

def grad_and_loss(func, argnum=0, method=False):
    """Return function that computes both gradient and loss value.

    :param func: The forward (loss) function.
    :param argnum: The index of argument to calculate gradient for.
    :param method: Skip the first argument 'self' if func is a method.
    :return: A function that would compute both the gradient of the specified argument and loss value.
    """
    # pylint: disable= missing-docstring
    @functools.wraps(func)
    def wrapped(*args):
        if method:
            args = args[1:]
        arrays = tuple(Value.wrap(a) for a in args)
        argnums = [argnum] if isinstance(argnum, int) else argnum
        for i in argnums:
            arrays[i]._marked_for_bp = True
        result_array = func(*arrays)
        _logger.debug('Forward pass finished. Start backward pass.')
        grad_vals = []
        for i in argnums:
            grad_vals.append(arrays[i].node.partial_derivative(
                result_array.node))
            arrays[i]._marked_for_bp = False
        if len(grad_vals) == 1:
            grad_vals = grad_vals[0]
        return grad_vals, result_array

    return wrapped
    # pylint: enable= missing-docstring


def grad(func, argnum=0, method=False):
    """Return function that contains gradient calculation.

    :param func: The forward (loss) function.
    :param argnum: The index of argument to calculate gradient for.
    :param method: Skip the first argument 'self' if func is a method.
    :return: A function that would compute the gradient of the specified argument.
    """
    grad_with_loss_func = grad_and_loss(func, argnum, method)
    # pylint: disable= missing-docstring

    @functools.wraps(grad_with_loss_func)
    def wrapped(*args):
        return grad_with_loss_func(*args)[0]

    return wrapped
    # pylint: enable= missing-docstring


class MXNetSymbolError(ValueError):
    """ Error class for creating mxnet symbols """
    pass

class Function(object):
    def __init__(self, symbol, input_shapes, name='mxnet_symbol'):
        """Construct a differentiable function from MXNet symbol.

        There is a known issue with current implementation. If SoftmaxLoss symbol is used, the
        ground truth label will be passed in in the forward. Therefore, even if no ground truth
        is provided during backward, the backward will "magically" run well since the required
        information has already been provided.

        :param symbol: Target symbol as function output.
        :param input_shapes: A dictionary of input names to input shapes, used for shape inference.
        :return: A function that could be called (and differentiated) as normal primitive.
        """
        self._symbol = symbol
        self._input_shapes = input_shapes
        self._sym_name = name
        self._executor, self._prim = self._create_prim()
        # Infer shapes of parameters and outputs.
        arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(**self._input_shapes)
        # Get shapes of learnable parameters.
        self._param_shapes = {}
        for i, arg_name in enumerate(symbol.list_arguments()):
            if arg_name not in input_shapes:
                self._param_shapes[arg_name] = arg_shapes[i]
        # Get shapes of output.
        self._out_shapes = {}
        for i, out_name in enumerate(symbol.list_outputs()):
            self._out_shapes[out_name] = out_shapes[i]
        # Get shapes of auxiliary tensors.
        self._aux_shapes = {}
        for i, aux_name in enumerate(symbol.list_auxiliary_states()):
            self._aux_shapes[aux_name] = aux_shapes[i]

    def _create_prim(self):
        dev = current_context().as_mxnet_context()
        executor = self._symbol.simple_bind(dev, 'write', **self._input_shapes)
        arg_names = self._symbol.list_arguments()
        # pylint: disable= missing-docstring
        # Define raw forward function.
        def func(**kwargs):
            # Set Data & Parameters
            for name, value in kwargs.items():
                if name in executor.arg_dict:
                    value.copyto(executor.arg_dict[name])
                else:
                    _logger.debug('Ignore unknown input (%s) to symbol (%s)' \
                            % (name, self._sym_name))
            # Forward computation.
            # TODO(haoran): How to set `is_train` flag
            executor.forward(is_train=True)
            # TODO(haoran): Currently doesn't support multiple outputs.
            return executor.outputs[0]
        # Set function name to be the given symbol name.
        func.__name__ = self._sym_name
        # Define gradient function generator.
        def gen_grad_kw(keyname):
            def grad_wrapper(ans, **kwargs):
                def grad_func(g):
                    executor.backward(out_grads=g)
                    ret = executor.grad_arrays[arg_names.index(keyname)]
                    return ret
                return grad_func
            return grad_wrapper
        # Create primitives.
        prim = Primitive(func, ArrayType.MXNET)
        for name in arg_names:
            prim.def_grad_kw(gen_grad_kw(name), name)
        return executor, prim
        # pylint: enable= missing-docstring

    def __call__(self, **kwargs):
        # Remove arguments that are not defined in symbol's argument
        # list.
        filtered_kwargs = {}
        for name, val in kwargs.items():
            if name in self._symbol.list_arguments():
                filtered_kwargs[name] = val
        return self._prim(**filtered_kwargs)

    def get_params(self):
        param_configs = {}
        for name, shape in self._param_shapes.items():
            param_configs[name] = { 'shape': shape }
        return param_configs

    def get_output_shapes(self):
        return self._out_shapes

    def get_one_output_shape(self):
        assert(len(self._out_shapes) == 1)
        return list(self._out_shapes.values())[0]

class MinpyWrapperError(TypeError):
    """ Error when wrapping function return values """
    pass


def _numpy_to_minpy(var):
    """ Convert a numpy array to minpy array """
    return Value.wrap(var)


def _minpy_to_numpy(var):
    """ Convert a minpy array to numpy array """
    return Value.wrap(var).get_data(ArrayType.NUMPY)


def numpy_to_minpy(var):
    """ Convert numpy array(s) to minpy array(s)

    :param var: singular, list, or tuple of numpy array(s)
    :return: singular, list, or tuple of minpy array(s)
    """
    if isinstance(var, (tuple, list)):
        return type(var)(Value.wrap(x) for x in var)
    else:
        return Value.wrap(var)


def minpy_to_numpy(var):
    """ Convert a minpy array to numpy array

    :param var: singular, list, or tuple of minpy array(s)
    :return: singular, list, or tuple of numpy array(s)
    """
    if isinstance(var, (tuple, list)):
        return type(var)(Value.wrap(x).get_data(ArrayType.NUMPY)
                         for x in var)
    else:
        return Value.wrap(var).get_data(ArrayType.NUMPY)


def convert(val, converter, basic_types):
    """Apply converter to the value according to their types.

    :param val: Value that could be either array types or container types.
    :param converter: A function to convert values.
    :param basic_types: Allowed types for conversion.
    :return: Converted value remained in its original contrainer structure.
    """
    if val is None:
        return None
    for ty in basic_types:
        if isinstance(val, ty):
            return converter(val)
    if isinstance(val, tuple):
        return tuple(convert(v, converter, basic_types) for v in val)
    elif isinstance(val, list):
        return list(convert(v, converter, basic_types) for v in val)
    elif isinstance(val, dict):
        return {k: convert(v, converter, basic_types) for k, v in val.items()}
    else:
        return val  # no conversion


def wraps(mode='lazy', method=False):
    """Convenient wrapper function separate MinPy and NumPy data structure.

    The wrapper will convert all array types in the input arguments as MinPy arrays.
    The return type will be converted according to the mode that is given.

    * In ``lazy`` mode, no conversion will be performed for the return values. So users need to
      handle the return value type themselves.
    * In ``numpy`` mode, all MinPy arrays will be converted to NumPy arrays.

    :param mode: the mode how wrapper performs on the return values
    :param method: set it to True only if the wrapper is applied on a method
    """
    # TODO: Refactor to let this decorator work without method option
    #pylint: disable= missing-docstring
    def wrapper(func):
        @functools.wraps(func)
        def real_wrapper(*args, **kwargs):
            basic_types = list(array_types.values())
            for num_type_lists in number_types.values():
                basic_types += num_type_lists
            basic_types += [Value]
            # convert input arguments into minpy structure
            if method:
                self = args[0]
                args = args[1:]

            mpy_args = convert(args, _numpy_to_minpy, basic_types)
            mpy_kwargs = convert(kwargs, _numpy_to_minpy, basic_types)

            if method:
                mpy_args = (self, ) + mpy_args
            # call func
            mpy_res = func(*mpy_args, **mpy_kwargs)
            # convert return value
            if mode == 'lazy':
                # lazy mode returns funciton result without converting
                return mpy_res
            elif mode == 'numpy':
                # convert every returned array to numpy.ndarray
                return convert(mpy_res, _minpy_to_numpy, basic_types)
            else:
                raise MinpyWrapperError('Unknown wrapper mode: %s' % mode)

        return real_wrapper
    # pylint: enable= missing-docstring
    return wrapper
