#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= protected-access, invalid-name
"""Core gradient calculation."""
from __future__ import absolute_import
from __future__ import print_function

import functools

from .array_variants import ArrayType
from .context import current_context
from .primitive import Primitive
from .utils import log
from . import tape
from . import array

_logger = log.get_logger(__name__)


def grad_and_loss(func, argnum=0):
    """Return function that computes both gradient and loss value.

    Parameters
    ----------
    func
        The forward (loss) function.
    argnum
        The index of argument to calculate gradient for.

    Returns
    -------
    function
        A function that would compute both the gradient of the specified argument and loss value.
    """

    @functools.wraps(func)
    def wrapped(*args):
        """Wrapped function."""
        arrays = tuple(array.wrap(a) for a in args)
        argnums = [argnum] if isinstance(argnum, int) else argnum
        with tape.tape() as current_tape:
            current_tape.start_recording()
            for i in argnums:
                arrays[i].mark_for_bp(current_tape)
            result = func(*arrays)
            current_tape.stop_recording()
            # TODO(minjie): wait for result value to be finished. This prevents
            # backward propagation to be run in parallel with forward propagation.
            # The main reason this is not allowed right now is due to the potential
            # large memory consumption. This should be fixed by a more systemetic
            # way in the future.
            if isinstance(result, array.Value):
                result.wait_to_read()
            else:
                for rst in result:
                    rst.wait_to_read()
            _logger.debug('Forward pass finished. Start backward pass.')
            # Get gradient using result as target.
            grad_vals = current_tape.get_gradient(
                tuple(arrays[i] for i in argnums), result)
            if len(grad_vals) == 1:
                grad_vals = grad_vals[0]
        return grad_vals, result

    return wrapped


def grad(func, argnum=0):
    """Return function that contains gradient calculation.

    Parameters
    ----------
    func
        The forward (loss) function.
    argnum
        The index of argument to calculate gradient for.

    Returns
    -------
    A function that would compute the gradient of the specified argument.
    """
    grad_with_loss_func = grad_and_loss(func, argnum)
    # pylint: disable= missing-docstring
    @functools.wraps(grad_with_loss_func)
    def wrapped(*args):
        return grad_with_loss_func(*args)[0]

    return wrapped
    # pylint: enable= missing-docstring


class MXNetSymbolError(ValueError):
    """Error class for creating mxnet symbols"""
    pass

class Function(object):
    """Container for MXNet symbol"""
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
        arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(
            **self._input_shapes)
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
        def func(*args):
            # Set Data & Parameters
            for arg, executor_arg in zip(args, executor.arg_arrays):
                if arg is not None:
                    arg.copyto(executor_arg)
            # Forward computation.
            # TODO(haoran): How to set `is_train` flag
            executor.forward(is_train=True)
            # TODO(haoran): Currently doesn't support multiple outputs.
            return executor.outputs[0]
        # Set function name to be the given symbol name.
        func.__name__ = self._sym_name

        # Define gradient function generator.
        def grad_wrapper(ans, *args): # pylint: disable= unused-argument
            def grad_func(g):
                executor.backward(out_grads=g)
                ret = executor.grad_arrays
                return ret

            return grad_func

        # Create primitives.
        prim = Primitive(func, ArrayType.MXNET)
        prim.def_multiple_grad(grad_wrapper, tuple(range(len(arg_names))))
        return executor, prim
        # pylint: enable= missing-docstring

    def __call__(self, **kwargs):
        # Remove arguments that are not defined in symbol's argument
        # list.
        ordered_args = [(kwargs[name] if name in kwargs else None)
                        for name in self._symbol.list_arguments()]
        return self._prim.call(args=ordered_args, kwargs={})

    # pylint: disable= missing-docstring
    def get_params(self):
        param_configs = {}
        for name, shape in self._param_shapes.items():
            param_configs[name] = {'shape': shape}
        return param_configs

    def get_output_shapes(self):
        return self._out_shapes

    def get_one_output_shape(self):
        assert (len(self._out_shapes) == 1)
        return list(self._out_shapes.values())[0]
    # pylint: enable= missing-docstring

def numpy_to_minpy(var):
    """Convert NumPy array(s) to MinPy array(s)

    Parameters
    ----------
    var
        singular, list, tuple, or dict of NumPy array(s)

    Returns
    -------
    singular, list, tuple, or dict of MinPy array(s)
    """
    return array.wrap(var)


def minpy_to_numpy(var):
    """Convert a MinPy array to NumPy array

    Parameters
    ----------
    var
        singular, list, tuple, or dict of MinPy array(s)

    Returns
    -------
    singular, list, tuple, or dict of NumPy array(s)
    """
    if isinstance(var, (tuple, list)):
        return type(var)(minpy_to_numpy(x) for x in var)
    elif isinstance(var, dict):
        return {k: minpy_to_numpy(v) for k, v in var.items()}
    else:
        return array.wrap(var).get_data(ArrayType.NUMPY)


def convert_args(func):
    """A wrapper that converts NumPy values into MinPy values (and leave MinPy
    values intact).

    Parameters
    ----------
    func
        A function

    Returns
    -------
    wrapper
        A wrapped function with all arguments converted to MinPy compatible types
    """
    # pylint: disable= missing-docstring
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # convert input arguments into Value
        mpy_args = tuple(array.wrap(a) for a in args)
        mpy_kwargs = {k: array.wrap(a) for k, a in kwargs.items()}
        # call func
        return func(*mpy_args, **mpy_kwargs)
    # pylint: enable= missing-docstring
    return wrapper


def return_numpy(func):
    """A wrapper that converts returns into NumPy values

    Parameters
    ----------
    func
        A function returns MinPy types

    Returns
    -------
    wrapper
        A wrapped function with returns converted to NumPy types
    """
    # pylint: disable= missing-docstring
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return minpy_to_numpy(func(*args, **kwargs))
    # pylint: enable= missing-docstring
    return wrapper
