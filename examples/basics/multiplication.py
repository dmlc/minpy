import minpy.numpy as np
import minpy.numpy.random as random
from minpy import core


def f(x, y):
    return np.dot(x, y)


def main():
    x = random.rand(3, 3)
    y = random.rand(3, 3)
    print('x: {}'.format(x.asnumpy()))
    print('y: {}'.format(y.asnumpy()))
    g = core.grad(f)
    print('grad: {}'.format(g(x, y).asnumpy()))


if __name__ == '__main__':
    main()
