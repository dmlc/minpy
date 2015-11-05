import functools
import numpy as np

class Node(object):

    def __init__(self, val):
        self._val = val
        self._partial_derivatives = []

    def __str__(self):
        return str(self._val)

    def add_partial_derivative(self, func, res):
        self._partial_derivatives.append((func, res))

    def partial_derivative(self, target):
        if self is target:
            return np.ones(self._val.shape)
        else:
            return functools.reduce(np.add, (map(lambda x: x[0](x[1].partial_derivative(target)), self._partial_derivatives)))

class Primitive(object):
    def __init__(self, func):
        self._func = func
        self._grad_func = {}

    def __call__(self, *args):
        arg_values = list(map(lambda x: x._val, args))
        result = self._func(*arg_values)
        result_node = Node(result)
        for i, arg in enumerate(args):
            arg.add_partial_derivative(self._grad_func[i](result, *arg_values), result_node)
        return result_node

    def def_grad(self, func, argnum=0):
        self._grad_func[argnum] = func

@Primitive
def mult(a, b):
    return a * b
mult.def_grad((lambda ans, x, y: lambda g: g * y), argnum=0)
mult.def_grad((lambda ans, x, y: lambda g: g * x), argnum=1)


if __name__ == '__main__':
    a = np.random.random((2, 2))
    a_node = Node(a)
    b = np.random.random((2, 2))
    b_node = Node(b)
    c = np.random.random((2, 2))
    c_node = Node(c)
    intermediate_node = mult(a_node, b_node)
    result_node = mult(intermediate_node, c_node)
    print(a_node.partial_derivative(result_node) == b * c)
    print(b_node.partial_derivative(result_node) == a * c)
    print(c_node.partial_derivative(result_node) == a * b)
