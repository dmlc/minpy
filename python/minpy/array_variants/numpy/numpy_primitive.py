# TODO everything
class Primitive(object):
    """Primitive computation."""
    __slots__ = ['_func', '_grad_func', '_grad_func_kw', '_type']

    def __init__(self, func):
        """Initialize.

        Args:
            func: A function that performs the action.
        """
        self._func = func
        self._grad_func = {}
        self._grad_func_kw = {}
        self._type = None  # will be set later by registry

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
            func: Gradient function.
            argnum: Index of the argument.

        Return:
            self instance for multiple def_grad in one statement
        """
        self._grad_func[argnum] = func
        return self

    def def_grad_kw(self, func, key):
        """Define gradient function.

        Args:
            func: Gradient function.
            key: Key name of the argument.
        """
        self._grad_func[key] = func

    def def_grad_zero(self, argnum=0):
        self._grad_func[argnum] = lambda *args, **kwargs: lambda g: 0.0
