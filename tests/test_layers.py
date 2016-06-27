import numpy as np
import mxnet as mx
from minpy import core
import minpy.nn.layers as layers
import minpy.utils.gradient_checker as gradient_checker
import minpy.dispatch.policy as plc

rng = np.random.RandomState(42)

def test_affine():
    x = rng.randn(20, 10)
    b = rng.randn(20, 1)
    fake_y = np.zeros([20, 5])
    def check_fn(w):
        return layers.l2_loss(layers.affine(x, w, b), fake_y)
    w = rng.randn(10, 5)
    gradient_checker.quick_grad_check(check_fn, w, rs=rng)

def test_relu():
    fake_y = np.zeros([2, 5])
    def check_fn(x):
        return layers.l2_loss(layers.relu(x), fake_y)
    x = rng.randn(2, 5)
    gradient_checker.quick_grad_check(check_fn, x, rs=rng)

def test_batchnorm():
    x = rng.randn(20, 40)
    gamma = rng.randn(1, 40)
    beta = rng.randn(1, 40)
    fake_y = np.zeros([20, 40])
    def check_gamma(g):
        y, _, _ = layers.batchnorm(x, g, beta)
        return layers.l2_loss(y, fake_y)
    gradient_checker.quick_grad_check(check_gamma, gamma, rs=rng)
    def check_beta(b):
        y, _, _ = layers.batchnorm(x, gamma, b)
        return layers.l2_loss(y, fake_y)
    gradient_checker.quick_grad_check(check_beta, beta, rs=rng)

def test_softmax():
    lbl = np.zeros([20])
    def check_fn(x):
        return layers.softmax_loss(x, lbl)
    x = rng.randn(20, 10)
    gradient_checker.quick_grad_check(check_fn, x, rs=rng)

def test_mxnet_affine():
    xshape = (10, 40)
    fake_y = np.zeros([10, 20])
    weights = rng.randn(20, 40) - 0.5
    inputs = mx.sym.Variable(name='x')
    fc = mx.sym.FullyConnected(name='fc', data=inputs, num_hidden=20)
    f = core.function(fc, [('x', xshape)])
    def check_fn(x):
        return layers.l2_loss(f(x = x, fc_weight = weights), fake_y)

    x = rng.randn(*xshape) - 0.5
    gradient_checker.quick_grad_check(check_fn, x, rs=rng)

def main():
    test_affine()
    test_relu()
    test_batchnorm()
    test_softmax()
    test_mxnet_affine()

if __name__ == '__main__':
    main()
