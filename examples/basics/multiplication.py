import logging

import minpy
import minpy.numpy as np
import minpy.numpy.random as random
from minpy import core

logging.getLogger('minpy.array').setLevel(logging.DEBUG)
logging.getLogger('minpy.core').setLevel(logging.DEBUG)
logging.getLogger('minpy.primitive').setLevel(logging.DEBUG)
minpy.set_global_policy('only_numpy')


def f(x, y):
    return np.multiply(x, y)


def main():
    x = random.rand(3, 3)
    y = random.rand(3, 3)
    print('x: {}'.format(x.asnumpy()))
    print('y: {}'.format(y.asnumpy()))
    g = core.grad(f, argnum=[0, 1])
    gr = g(x, y)
    print('grad_x: {}'.format(gr[0].asnumpy()))
    print('grad_y: {}'.format(gr[1].asnumpy()))


if __name__ == '__main__':
    main()
