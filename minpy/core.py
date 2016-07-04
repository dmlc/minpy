#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= protected-access, invalid-name
"""Core gradient calculation."""
from __future__ import absolute_import
from __future__ import print_function

import mxnet as mx
import functools

from .utils import log
from . import array
from .array_variants import ArrayType, array_types, number_types

_logger = log.get_logger(__name__)


def grad_and_loss(func, argnum=0):
    """Return function that computes both gradient and loss value.

    :param func: The forward (loss) function.
    :param argnum: The index of argument to calculate gradient for.
    :return: A function that would compute both the gradient of the specified argument and loss value.
    """
    # pylint: disable= missing-docstring
    @functools.wraps(func)
    def wrapped(*args):
        arrays = tuple(array.Value.wrap(a) for a in args)
        argnums = [argnum] if isinstance(argnum, int) else argnum
        for i in argnums:
            arrays[i]._marked_for_bp = True
        result_array = func(*arrays)
        _logger.debug('---Forward pass finished. Start backward pass')
        grad_vals = []
        for i in argnums:
            grad_vals.append(arrays[i].node.partial_derivative(result_array.node))
            arrays[i]._marked_for_bp = False
        if len(grad_vals) == 1:
            grad_vals = grad_vals[0]
        return grad_vals, result_array

    return wrapped
    # pylint: enable= missing-docstring


def grad(func, argnum=0):
    """Return function that contains gradient calculation.

    :param func: The forward (loss) function.
    :param argnum: The index of argument to calculate gradient for.
    :return: A function that would compute the gradient of the specified argument.
    """
    grad_with_loss_func = grad_and_loss(func, argnum)
    # pylint: disable= missing-docstring

    @functools.wraps(grad_with_loss_func)
    def wrapped(*args):
        return grad_with_loss_func(*args)[0]
    return wrapped
    # pylint: enable= missing-docstring


class MXNetSymbolError(ValueError):
    """ Error class for creating mxnet symbols """
    pass


def function(symbol, input_shapes, sym_name='mxnet_symbol'):
    """Construct a differentiable function from MXNet symbol.

    :param symbol: Target symbol as function output.
    :param input_shapes: A dictionary of input names to input shapes, used for shape inference.
    :return: A function that could be called (and differentiated) as normal primitive.
    """
    # TODO: Policy Control
    policy_cpu = False
    dev = mx.cpu() if policy_cpu else mx.gpu(int(0))
    dshape = {name: shape for name, shape in input_shapes.items()}
    executor = symbol.simple_bind(dev, 'write', **dshape)
    arg_names = symbol.list_arguments()
    """ In train model of mxnet example, there's no grad of input(data)
    While it has grad of input in Minpy's calling example
    Possible culprit: In model, training is complete, i.e. loss is computed in symbol. Not in Minpy.
    ```python
    input_names = dshape.keys()
    raw_param_names = list(set(arg_names) - set(input_names))
    raw_param_names = list(set(arg_names))
    param_idx = [i for i in range(len(arg_names)) if arg_names[i] in raw_param_names]
    param_names = [arg_names[i] for i in param_idx]
    ```
    """
    # pylint: disable= missing-docstring
    param_names = arg_names

    def func(*args, **kwargs):
        if len(args) > 0:
            raise MXNetSymbolError('find arg with no name specified')
        # Set Data & Parameters
        for name, value in kwargs.items():
            if name in executor.arg_dict:
                value.copyto(executor.arg_dict[name])
            else:
                raise MXNetSymbolError(
                    'find arg name: %s not in executors arg_list' %
                    name)
        # forward
        # TODO: is_train flag
        executor.forward(is_train=True)
        # TODO: Minpy currently doesn't support multiple outputs
        return executor.outputs[0]
    func.__name__ = sym_name

    def def_grad_kw(keyname):
        def grad_wrapper(ans, *arg_values, **kwargs_values):
            def grad_func(g):
                executor.backward(out_grads=g)
                ret = executor.grad_arrays[param_names.index(keyname)]
                return ret
            return grad_func
        return grad_wrapper
    prim = array.Primitive(func, ArrayType.MXNET)
    for name in param_names:
        prim.def_grad_kw(def_grad_kw(name), name)
    return prim
    # pylint: enable= missing-docstring


class MinpyWrapperError(TypeError):
    """ Error when wrapping function return values """
    pass


def numpy_to_minpy(var):
    """ Convert a numpy array to minpy array """
    return array.Value.wrap(var)


def minpy_to_numpy(var):
    """ Convert a minpy array to numpy array """
    return array.Value.wrap(var).get_data(ArrayType.NUMPY)


def convert(val, converter, basic_types):
    """Apply converter to the value according to their types.

    :param val: Value that could be either array types or container types.
    :param converter: A function to convert values.
    :param basic_types: Allowed types for conversion.
    :return: Converted value remained in its original contrainer structure.
    """
    if val is None:
        return None
    ret = None
    if type(val) in basic_types:
        ret = converter(val)
    elif isinstance(val, tuple):
        ret = tuple(convert(v, converter, basic_types) for v in val)
    elif isinstance(val, list):
        ret = list(convert(v, converter, basic_types) for v in val)
    elif isinstance(val, dict):
        ret = {k: convert(v, converter, basic_types)
               for k, v in val.items()}
    else:
        raise MinpyWrapperError(
            'Unexpected %s type found in core.convert' %
            type(val))
    return ret


def wraps(mode='lazy'):
    """Convenient wrapper function separate MinPy and NumPy data structure.
    
    The wrapper will convert all array types in the input arguments as MinPy arrays.
    The return type will be converted according to the mode that is given.

    * In ``lazy`` mode, no conversion will be performed for the return values. So users need to 
      handle the return value type themselves.
    * In ``numpy`` mode, all MinPy arrays will be converted to NumPy arrays.
    """
    #pylint: disable= missing-docstring
    def wrapper(func):
        @functools.wraps(func)
        def real_wrapper(*args, **kwargs):
            basic_types = list(array_types.values())
            for num_type_lists in number_types.values():
                basic_types += num_type_lists
            basic_types += [array.Number, array.Array, array.Value]
            # convert input arguments into minpy structure
            mpy_args = convert(args, numpy_to_minpy, basic_types)
            mpy_kwargs = convert(kwargs, numpy_to_minpy, basic_types)
            # call func
            mpy_res = func(*mpy_args, **mpy_kwargs)
            # convert return value
            if mode == 'lazy':
                # lazy mode returns funciton result without converting
                return mpy_res
            elif mode == 'numpy':
                # convert every returned array to numpy.ndarray
                return convert(mpy_res, minpy_to_numpy, basic_types)
            else:
                raise MinpyWrapperError('Unknown wrapper mode: %s' % mode)
        return real_wrapper
    # pylint: enable= missing-docstring
    return wrapper
