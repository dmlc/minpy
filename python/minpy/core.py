#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core gradient calculation."""
import mxnet as mx
import functools
import operator
from .utils import log
from . import array

from .array_variants import ArrayType

_logger = log.get_logger(__name__)

def grad(func, argnum=0):
    @functools.wraps(func)
    def wrapped(*args):
        def make_array(x):
            return x if isinstance(x, array.Value) else array.Value.wrap(x)
        arrays = tuple(map(make_array, args))
        argnums = [argnum] if type(argnum) is int else argnum
        for i in argnums:
            arrays[i]._marked_for_bp = True
        result_array = func(*arrays)
        _logger.debug('Forward pass finished. Start backward pass.')
        grad_vals = []
        for i in argnums:
            grad_vals.append(arrays[i].node.partial_derivative(result_array.node))
            arrays[i]._marked_for_bp = False
        if len(grad_vals) == 1:
            grad_vals = grad_vals[0]
        return grad_vals
    return wrapped

def grad_and_loss(func, argnum=0):
    @functools.wraps(func)
    def wrapped(*args):
        def make_array(x):
            return x if isinstance(x, array.Value) else array.Value.wrap(x)
        arrays = tuple(map(make_array, args))
        argnums = [argnum] if type(argnum) is int else argnum
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

class MXNetSymbolArgErrorLackName(ValueError):
    pass

class MXNetSymbolArgErrorUnknownName(ValueError):
    pass

def function(symbol, input_shapes):
    # TODO: Policy Control
    policy_cpu = False
    dev = mx.cpu() if policy_cpu else mx.gpu(int(0))

    dshape = {name: shape for name, shape in input_shapes}

    executor = symbol.simple_bind(dev, 'write', **dshape)

    arg_names = symbol.list_arguments()
    """ In train model of mxnet example, there's no grad of input(data)
    While it has grad of input in Minpy's calling example
    Possible culprit: In model, training is complete, i.e. loss is computed in symbol. Not in Minpy.

        input_names = dshape.keys()
        raw_param_names = list(set(arg_names) - set(input_names))
        raw_param_names = list(set(arg_names))
        param_idx = [i for i in range(len(arg_names)) if arg_names[i] in raw_param_names]
        param_names = [arg_names[i] for i in param_idx]
    """

    param_names = arg_names

    def func(*args, **kwargs):
      
      if len(args) > 0:
        raise MXNetSymbolArgErrorLackName('find arg with no name specified')

      # Set Data & Parameters
      for name, value in kwargs.items():
        if name in executor.arg_dict:
          value.copyto(executor.arg_dict[name])
        else:
          raise MXNetSymbolArgErrorUnknownName('find arg name: %s not in executors arg_list' % name)

      # forward
      # TODO: is_train flag
      executor.forward(is_train=True)

      # TODO: Handle with multiple outputs, including the order of outputs 
      return executor.outputs[0]

    func.__name__ = 'mxnet_symbol'

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

def NumpyVarToMinpy(var):
  return array.Value.wrap(var)

def MinpyVarToNumpy(var):
  return array.Value.wrap(var).get_data(ArrayType.NUMPY)

def converter(cmd):
  def wrapper(func):
    @functools.wraps(func)
    def real_wrapper(*args, **kwargs):
      mpy_args = [NumpyVarToMinpy(v) for v in args]
      mpy_kwargs = {}
      for key, value in mpy_kwargs.iteritems():
        mpy_kwargs[key] = NumpyVarToMinpy(value)

      mpy_res = func(*mpy_args, **mpy_kwargs)
      if cmd == 'lazy':
        return mpy_res

      if type(mpy_res) is not tuple:
        loss_mpy = mpy_res
      else:
        loss_mpy = mpy_res[0]
      loss_npy = MinpyVarToNumpy(loss_mpy)

      if type(mpy_res) is not tuple:
        return loss_npy
      else:
        grad_dict_mpy = mpy_res[1]
        grad_dict_npy = {}
        for key, value in grad_dict_mpy.iteritems():
          grad_dict_npy[key] = MinpyVarToNumpy(value)
        return loss_npy, grad_dict_npy
    return real_wrapper
  return wrapper
