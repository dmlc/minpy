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
            # Wrap result value.
            # TODO(minjie): Also wait for result value to be finished. This prevents
            # backward propagation to be run in parallel with forward propagation.
            # The main reason this is not allowed right now is due to the potential
            # large memory consumption. This should be fixed by a more systemetic
            # way in the future.
            result_wrapped = None
            if isinstance(result, tuple):
                result_wrapped = tuple(array.wrap(rst) for rst in result)
            else:
                result_wrapped = array.wrap(result) # pylint: disable=redefined-variable-type
            _logger.debug('Forward pass finished. Start backward pass.')
            # Get gradient using result as target.
            grad_vals = current_tape.get_gradient(
                tuple(arrays[i] for i in argnums), result_wrapped)
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

    def __init__(self, symbol, input_shapes=None, name='mxnet_symbol'):
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
        self._is_train = True
        self._input_shapes = input_shapes
        if input_shapes is not None:
            self._infer_shape(input_shapes)
        self._sym_name = name

    @property
    def is_train(self):
        """ whether this forward is for evaluation purpose. """
        return self._is_train

    @is_train.setter
    def is_train(self, value):
        """ whether this forward is for evaluation purpose. """
        self._is_train = value

    def _infer_shape(self, input_shapes):
        # Infer shapes of parameters and outputs.
        arg_shapes, out_shapes, aux_shapes = self._symbol.infer_shape(
            **self._input_shapes)
        # Get shapes of learnable parameters.
        self._param_shapes = {}
        for i, arg_name in enumerate(self._symbol.list_arguments()):
            if arg_name not in input_shapes:
                self._param_shapes[arg_name] = arg_shapes[i]
        # Get shapes of output.
        self._out_shapes = {}
        for i, out_name in enumerate(self._symbol.list_outputs()):
            self._out_shapes[out_name] = out_shapes[i]
        # Get shapes of auxiliary tensors.
        self._aux_shapes = {}
        for i, aux_name in enumerate(self._symbol.list_auxiliary_states()):
            self._aux_shapes[aux_name] = aux_shapes[i]

    def _create_prim(self, global_dict):
        dev = current_context().as_mxnet_context()
        executor = self._symbol.simple_bind(dev, 'write', **self._input_shapes)
        arg_names = self._symbol.list_arguments()
        aux_state_names = self._symbol.list_auxiliary_states()

        # pylint: disable= missing-docstring
        # Define raw forward function.
        def func(*args):
            # Set Data & Parameters
            for arg, executor_arg in zip(args[:len(arg_names)], executor.arg_arrays):
                if arg is not None:
                    arg.copyto(executor_arg)

            for aux, executor_aux in zip(args[len(arg_names):], executor.aux_arrays):
                if aux is not None:
                    aux.copyto(executor_aux)

            # Forward computation.
            executor.forward(is_train=self._is_train)
            return tuple(executor.outputs) if len(executor.outputs) > 1 else executor.outputs[0]
        # Set function name to be the given symbol name.
        func.__name__ = self._sym_name

        # Define gradient function generator.
        def grad_wrapper(ans, *args): # pylint: disable= unused-argument
            def grad_func(g):
                executor.backward(out_grads=g)

                for aux, executor_aux in zip(global_dict['aux_states'], executor.aux_arrays):
                    if aux is not None:
                        aux[:] = executor_aux

                ret = executor.grad_arrays
                ret += [0] * len(aux_state_names)
                return ret

            return grad_func

        # Create primitives.
        prim = Primitive(func, ArrayType.MXNET)
        prim.def_multiple_grad(grad_wrapper, tuple(range(len(arg_names) + len(aux_state_names))))
        return prim
        # pylint: enable= missing-docstring

    def __call__(self, **kwargs):
        if self._input_shapes is None:
            self._infer_shape({kv[0]: kv[1].shape for kv in kwargs.items()})
        # Remove arguments that are not defined in symbol's argument
        # list.
        ordered_args = [(kwargs[name] if name in kwargs else None)
                        for name in self._symbol.list_arguments()]
        ordered_aux_states = [(kwargs[name] if name in kwargs else None)
                        for name in self._symbol.list_auxiliary_states()]
        prim = self._create_prim({'aux_states' : ordered_aux_states})
        return prim.call(args=ordered_args + ordered_aux_states, kwargs={})

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
