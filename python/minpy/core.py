from __future__ import absolute_import

import functools
import operator
from .utils import common
from .utils import log

logger = log.get_logger(__name__, log.logging.WARNING)

autograd_mode = False

class AutogradMode:
    def __init__(self):
        pass
    def __enter__(self):
        global autograd_mode
        autograd_mode = True
    def __exit__(slef, type, value, traceback):
        global autograd_mode
        autograd_mode = False

value_type_to_node_type = dict()

def register_node_type(value_type, node_type):
    value_type_to_node_type[value_type] = node_type

def wrap(x):
    if isinstance(x, Node):
        # if already a node type, do not need to wrap it again
        return x
    elif type(x) in value_type_to_node_type:
        # if registered a specific node type, wrap it with the registered type
        node_type = value_type_to_node_type[type(x)]
        return node_type(x)
    else:
        # if nothing specified, wrap it with the base Node type
        return Node(x)

def unwrap(x):
    return x._val if isinstance(x, Node) else x

class Node(object):
    """Node that wraps a value (numpy.ndarray, mxnet.ndarray, etc.)."""

    def __init__(self, val):
        """Initialize.

        Args:
            val: Value to wrap.
        """
        self._val = val
        self._partial_derivatives = []
        self._partial_derivative_cache = {}

    def __str__(self):
        """Get string representation.

        Return:
            A string representation.
        """
        return 'Node({})'.format(self._val)

    @property
    def val(self):
        return self._val

    def add_partial_derivative(self, func, res):
        logger.info('Adding partial derivative to {}: {}'.format(id(self), self))
        self._partial_derivatives.append((func, res))

    def partial_derivative(self, target):
        if target in self._partial_derivative_cache:
            return self._partial_derivative_cache[target]
        else:
            if self is target: # partial derivative of self is one
                return 1.0  # TODO shape? 
            else:
                res = functools.reduce(operator.add, map(
                    lambda x: x[0](x[1].partial_derivative(target)),
                    self._partial_derivatives), 0.0) # TODO shape?
                self._partial_derivative_cache[target] = res
                logger.info('Partial derivative id: {}, shape: {}, value: {}'.format(
                    id(self), self.val.shape, res))
                return res

class Primitive(object):
    """Primitive computation."""

    def __init__(self, func):
        """Initialize.

        Args:
            func: A function that performs the action.
        """
        self._func = func
        self._grad_func = {}
        self._grad_func_kw = {}

    def __call__(self, *args, **kwargs):
        """Call wrapped function.

        Args:
            *args: Arguments for the wrapped function.
            **kwargs: Arguments for the wrapped function.

        Returns:
            A `Node` representing the result.

        Raises:
            IndexError: No corresponding gradient function.
            KeyError: No corresponding gradient function.
        """
        logger.info('Calling {}'.format(self._func))
        def get_val(x):
            return unwrap(x._val if isinstance(x, Node) else x)
        # unwrap Node or Wrapper to get underlying value
        arg_values = tuple(map(get_val, args))
        kwargs_values = {x: get_val(kwargs[x]) for x in kwargs}
        # call the real function with raw value
        result_value = self._func(*arg_values, **kwargs_values)
        # wrap the result raw value with wrapper and node
        if autograd_mode:
            result = wrap(result_value)
            # record partial derivative paths
            for i, arg in enumerate(args):
                if isinstance(arg, Node):
                    arg.add_partial_derivative(self._grad_func[i](
                        result_value, *arg_values, **kwargs_values), result)
            for x in kwargs:
                if isinstance(arg, Node):
                    arg.add_partial_derivative(self._grad_func_kw[x](
                        result_value, *arg_values, **kwargs_values), result)
        else:
            result = result_value

        return result

    def def_grad(self, func, argnum=0):
        """Define gradient function.

        Args:
            func: Gradient function.
            argnum: Index of the argument.
        """
        self._grad_func[argnum] = func

    def def_grad_kw(self, func, key):
        """Define gradient function.

        Args:
            func: Gradient function.
            key: Key name of the argument.
        """
        self._grad_func[key] = func

    def def_grad_zero(self, argnum=0):
        self._grad_func[argnum] = lambda *args, **kwargs: lambda g: 0.0 # TODO shape?

def grad(func, argnum=0):
    @functools.wraps(func)
    def wrapped(*args):
        with AutogradMode(): # start autograd mode
            nodes = tuple(map(wrap, args)) # first wrap the input args with Node type
            result_node = func(*nodes)
        return nodes[argnum].partial_derivative(result_node)
    return wrapped
