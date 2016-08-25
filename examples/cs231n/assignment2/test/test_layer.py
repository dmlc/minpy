from cs231n.layers import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
import time

import numpy as np

import minpy
import minpy.numpy as minpy_np
import minpy.core
import minpy.array
from minpy.array_variants import ArrayType
import minpy.dispatch.policy as policy


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


data = get_CIFAR10_data()
for k, v in data.iteritems():
    print '%s: ' % k, v.shape


def NumpyVarToMinpy(var):
    return minpy.array.Value.wrap(var)


def MinpyVarToNumpy(var):
    return minpy.array.Value.wrap(var).get_data(ArrayType.NUMPY)


def Test_Affine_Forward():
    # Test the affine_forward function
    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(
        np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    mp_x = NumpyVarToMinpy(x)
    mp_w = NumpyVarToMinpy(w)
    mp_b = NumpyVarToMinpy(b)
    mp_out, _ = affine_forward(x, w, b)
    out = MinpyVarToNumpy(mp_out)

    correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                            [3.25553199, 3.5141327, 3.77273342]])

    # Compare your output with ours. The error should be around 1e-9.
    print 'Testing affine_forward function:'
    print 'difference: ', rel_error(out, correct_out)


def Test_Affine_Backward():
    # Test the affine_backward function
    np.random.seed(123)
    x = np.random.randn(10, 6)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0].asnumpy(), x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0].asnumpy(), w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0].asnumpy(), b, dout)

    _, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)

    # The error should be around 1e-9
    print 'Testing affine_backward function:'
    print 'dx error: ', rel_error(dx_num, dx.asnumpy())
    print 'dw error: ', rel_error(dw_num, dw.asnumpy())
    print 'db error: ', rel_error(db_num, db.asnumpy())

def Relu_Forward():
    # Test the relu_forward function
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
    mp_x = NumpyVarToMinpy(x)

    mp_out, _ = relu_forward(mp_x)
    out = MinpyVarToNumpy(mp_out)

    correct_out = np.array([[0.,
                             0.,
                             0.,
                             0.,], [0.,
                                    0.,
                                    0.04545455,
                                    0.13636364,], [0.22727273,
                                                   0.31818182,
                                                   0.40909091,
                                                   0.5,]])

    # Compare your output with ours. The error should be around 1e-8
    print 'Testing relu_forward function:'
    print 'difference: ', rel_error(out, correct_out)


def Relu_Backward():
    np.random.seed(123)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(
        lambda x: MinpyVarToNumpy(relu_forward(NumpyVarToMinpy(x))[0]), x, dout)

    mp_x = NumpyVarToMinpy(x)
    mp_dout = NumpyVarToMinpy(dout)
    _, cache = relu_forward(mp_x)
    mp_dx = relu_backward(mp_dout, cache)
    dx = MinpyVarToNumpy(mp_dx)

    # The error should be around 1e-12
    print 'Testing relu_backward function:'
    print 'dx', dx
    print 'dx_num', dx
    print 'dx error: ', rel_error(dx_num, dx)


def Test_Test_Forward():
    test_sum_forward()

def Test_Softmax():
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    mp_x = NumpyVarToMinpy(x)
    mp_y = NumpyVarToMinpy(y)
    #dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
    mp_loss, mp_dx = softmax_loss(mp_x, mp_y)

    loss = MinpyVarToNumpy(mp_loss)
    dx = MinpyVarToNumpy(mp_dx)

    # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
    print 'Testing softmax_loss:'
    print 'loss: ', loss
    #print 'dx error: ', rel_error(dx_num, dx)


def Test_Dropout():
    x = np.random.randn(10, 10) + 10
    mp_x = NumpyVarToMinpy(x)

    dout = np.random.randn(*x.shape)
    p = 0.5

    dropout_param = {'mode': 'train', 'p': p, 'seed': 123}

    mp_out = dropout_forward(mp_x, dropout_param)
    out = MinpyVarToNumpy(mp_out)

    print('Probability:', p)
    print('Filterd ratio:', (out == 0).sum() / 100.0)

def Test_Dropout_backward():
    x = np.random.randn(10, 10) + 10
    dout = np.random.randn(*x.shape)
    
    dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}
    out, cache = dropout_forward(x, dropout_param)
    dx_num = eval_numerical_gradient_array(lambda x: dropout_forward(x, dropout_param)[0].asnumpy(), x, dout)
    dx = dropout_backward(dout, cache)
    
    print 'Testing affine_relu_forward:'
    print 'dx error: ', rel_error(dx_num, dx.asnumpy())

def Test_BatchNorm():
    # Check the test-time forward pass by running the training-time
    # forward pass many times to warm up the running averages, and then
    # checking the means and variances of activations after a test-time
    # forward pass.
    
    N, D1, D2, D3 = 200, 50, 60, 3
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    
    bn_param = {'mode': 'train'}
    gamma = np.ones(D3)
    beta = np.zeros(D3)
    for t in xrange(5):
      X = np.random.randn(N, D1)
      a = np.maximum(0, X.dot(W1)).dot(W2)
      batchnorm_forward(a, gamma, beta, bn_param)
      print bn_param
    bn_param['mode'] = 'test'
    X = np.random.randn(N, D1)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    a_norm = batchnorm_forward(a, gamma, beta, bn_param)[0].asnumpy()
    
    
    # Means should be close to zero and stds close to one, but will be
    # noisier than training-time forward passes.
    print 'After batch normalization (test-time):'
    print '  means: ', a_norm.mean(axis=0)
    print '  stds: ', a_norm.std(axis=0)

def Test_BatchNorm_backward():
    N, D = 100, 500
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)
    
    bn_param = {'mode': 'train'}
    out, _, _, cache = batchnorm_forward(x, gamma, beta, None, None, bn_param)
    
    t1 = time.time()
    dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)
    t2 = time.time()
    dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)
    t3 = time.time()
    
    print 'dx difference: ', rel_error(dx1.asnumpy(), dx2.asnumpy())
    print 'dgamma difference: ', rel_error(dgamma1.asnumpy(), dgamma2.asnumpy())
    print 'dbeta difference: ', rel_error(dbeta1.asnumpy(), dbeta2.asnumpy())
    print 'speedup: %.2fx' % ((t2 - t1) / (t3 - t2))

def Test_connect_layer():
    x = np.random.randn(2, 12)
    w = np.random.randn(12, 10)
    b = np.random.randn(10)
    dout = np.random.randn(2, 10)
    
    out, cache = affine_relu_forward(x, w, b)
    dx, dw, db = affine_relu_backward(dout, cache)
    
    dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0].asnumpy(), x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0].asnumpy(), w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0].asnumpy(), b, dout)
    
    print 'Testing affine_relu_forward:'
    print 'dx error: ', rel_error(dx_num, dx.asnumpy())
    print 'dw error: ', rel_error(dw_num, dw.asnumpy())
    print 'db error: ', rel_error(db_num, db.asnumpy())

def Test_loss():
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)
    
    dx_num = eval_numerical_gradient(lambda x: svm_loss_forward(x, y)[0].asnumpy(), x, verbose=False)
    loss, cache = svm_loss_forward(x, y)
    dx = svm_loss_backward(cache)
    
    # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
    print 'Testing svm_loss:'
    print 'loss: ', loss
    print 'dx error: ', rel_error(dx_num, dx.asnumpy())
    
    dx_num = eval_numerical_gradient(lambda x: softmax_loss_forward(x, y)[0].asnumpy(), x, verbose=False)
    loss, cache = softmax_loss_forward(x, y)
    dx = softmax_loss_backward(cache)
    
    # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
    print '\nTesting softmax_loss:'
    print 'loss: ', loss
    print 'dx error: ', rel_error(dx_num, dx.asnumpy())

Test_Affine_Forward()
Test_Affine_Backward()
Relu_Forward()
Relu_Backward()
Test_connect_layer()
Test_loss()
Test_Dropout_backward()
Test_SVM()
Test_SVM_CPU_GPU()
Test_Softmax()
Test_Test_Forward()
Test_Dropout()
Test_BatchNorm()
Test_BatchNorm_backward()
