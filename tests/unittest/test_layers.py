import sys
import numpy as np
import mxnet as mx
import minpy.numpy as mp
from minpy import core
import minpy.nn.layers as layers
import minpy.utils.gradient_checker as gradient_checker
import minpy.dispatch.policy as plc

rng = np.random.RandomState(42)

def test_layers():
    
    def test_basic():
        @core.convert_args
        def check_fn(x):
            return mp.sum(x + x)
        x = rng.randn(10, 5)
        return gradient_checker.quick_grad_check(check_fn, x, rs=rng)

    def test_forward_unused():
        @core.convert_args
        def check_fn(x):
            y = x + 1
            print(mp.exp(y))
            return mp.sum(2 * y)
        x = rng.randn(10, 5)
        return gradient_checker.quick_grad_check(check_fn, x, rs=rng)

    def test_affine():
        x = rng.randn(20, 10)
        b = rng.randn(20, 1)
        fake_y = np.zeros([20, 5])
        fake_y[:,0] = 1
    
        @core.convert_args
        def check_fn(w):
            return layers.softmax_loss(layers.affine(x, w, b), fake_y)
    
        w = rng.randn(10, 5)
        return gradient_checker.quick_grad_check(check_fn, w, rs=rng)
    
    
    def test_relu():
        fake_y = np.zeros([2, 5])
        fake_y[:,0] = 1
    
        @core.convert_args
        def check_fn(x):
            return layers.softmax_loss(layers.relu(x), fake_y)
    
        x = rng.randn(2, 5)
        return gradient_checker.quick_grad_check(check_fn, x, rs=rng)
    
    
    def test_batchnorm():
        x = rng.randn(20, 40)
        gamma = rng.randn(1, 40)
        beta = rng.randn(1, 40)
        fake_y = np.zeros([20, 40])
        fake_y[:,0] = 1
    
        @core.convert_args
        def check_gamma(g):
            y, _, _ = layers.batchnorm(x, g, beta)
            return layers.softmax_loss(y, fake_y)
    
        @core.convert_args
        def check_beta(b):
            y, _, _ = layers.batchnorm(x, gamma, b)
            return layers.softmax_loss(y, fake_y)
    
        return (gradient_checker.quick_grad_check(check_gamma, gamma, rs=rng)
                and gradient_checker.quick_grad_check(check_beta, beta, rs=rng))
    
    
    def test_softmax():
        lbl = np.zeros([20])
    
        @core.convert_args
        def check_fn(x):
            return layers.softmax_loss(x, lbl)
    
        x = rng.randn(20, 10)
        return gradient_checker.quick_grad_check(check_fn, x, rs=rng)
    
    
    def test_mxnet_affine():
        xshape = (10, 40)
        fake_y = np.zeros([10, 20])
        fake_y[:,0] = 1
        x = rng.randn(*xshape)
    
        inputs = mx.sym.Variable(name='x')
        fc = mx.sym.FullyConnected(name='fc', data=inputs, num_hidden=20)
        f = core.Function(fc, {'x': xshape})
    
        @core.convert_args
        def check_fn(weights):
            return layers.softmax_loss(f(x=x, fc_weight=weights), fake_y)
        weights = rng.randn(20, 40) * 0.01
    
        return gradient_checker.quick_grad_check(check_fn, weights, rs=rng)
    
    
    def test_mxnet_softmax():
        xshape = (10, 40)
        fake_y = np.zeros([10,])
        x = rng.randn(*xshape)
    
        net = mx.sym.Variable(name='x')
        net = mx.sym.FullyConnected(net, name='fc', num_hidden=20)
        net = mx.sym.SoftmaxOutput(net, name='softmax', normalization='batch')
        f = core.Function(net, {'x': xshape, 'softmax_label': fake_y.shape})
    
        @core.convert_args
        def check_fn(weights):
            return layers.softmax_cross_entropy(
                    f(x=x, softmax_label=fake_y, fc_weight=weights),
                    fake_y)
        weights = rng.randn(20, 40) * 0.01
    
        return gradient_checker.quick_grad_check(check_fn, weights, rs=rng)

    return (test_basic()
            and test_forward_unused()
            and test_affine()
            and test_relu()
            and test_batchnorm()
            and test_softmax()
            and test_mxnet_affine()
            and test_mxnet_softmax())

if __name__ == '__main__':
    sys.exit(not test_layers())
