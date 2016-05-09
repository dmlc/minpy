from cs231n.layers_test import *
from cs231n.layer_utils_test import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

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
  w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
  b = np.linspace(-0.3, 0.1, num=output_dim)
  
  mp_x = NumpyVarToMinpy(x)
  mp_w = NumpyVarToMinpy(w)
  mp_b = NumpyVarToMinpy(b)
  mp_out, _ = affine_forward(x, w, b)
  out = MinpyVarToNumpy(mp_out)

  correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                            [ 3.25553199,  3.5141327,   3.77273342]])
  
  # Compare your output with ours. The error should be around 1e-9.
  print 'Testing affine_forward function:'
  print 'difference: ', rel_error(out, correct_out)

def Test_Affine_Backward():
  # Test the affine_backward function
  np.random.seed(123)
  x = np.random.randn(10, 2, 3)
  w = np.random.randn(6, 5)
  b = np.random.randn(5)
  dout = np.random.randn(10, 5)

  mp_x = NumpyVarToMinpy(x)
  mp_w = NumpyVarToMinpy(w)
  mp_b = NumpyVarToMinpy(b)
  
  dx_num = eval_numerical_gradient_array(lambda x: MinpyVarToNumpy(affine_forward(NumpyVarToMinpy(x), mp_w, mp_b)[0]), x, dout)
  dw_num = eval_numerical_gradient_array(lambda w: MinpyVarToNumpy(affine_forward(NumpyVarToMinpy(x), mp_w, mp_b)[0]), w, dout)
  db_num = eval_numerical_gradient_array(lambda b: MinpyVarToNumpy(affine_forward(NumpyVarToMinpy(x), mp_w, mp_b)[0]), b, dout)

  _, cache = affine_forward(mp_x, mp_w, mp_b)
  mp_dx, mp_dw, mp_db = affine_backward(dout, cache)

  dx = MinpyVarToNumpy(mp_dx)
  dw = MinpyVarToNumpy(mp_dw)
  db = MinpyVarToNumpy(mp_db)
  
  # The error should be around 1e-10
  print 'Testing affine_backward function:'
  print 'dx: ', dx
  print 'dw: ', dw
  print 'db: ', db
  print 'dx error: ', rel_error(dx_num, dx)
  print 'dw error: ', rel_error(dw_num, dw)
  print 'db error: ', rel_error(db_num, db)


def Relu_Forward():
  # Test the relu_forward function
  x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
  mp_x = NumpyVarToMinpy(x)
  
  mp_out, _ = relu_forward(mp_x)
  out = MinpyVarToNumpy(mp_out)

  correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                          [ 0.,          0.,          0.04545455,  0.13636364,],
                          [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])
  
  # Compare your output with ours. The error should be around 1e-8
  print 'Testing relu_forward function:'
  print 'difference: ', rel_error(out, correct_out)

def Relu_Backward():
  np.random.seed(123)
  x = np.random.randn(10, 10)
  dout = np.random.randn(*x.shape)
  
  dx_num = eval_numerical_gradient_array(lambda x: MinpyVarToNumpy(relu_forward(NumpyVarToMinpy(x))[0]), x, dout)
  
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

def Test_SVM():
  np.random.seed(31)
  num_classes, num_inputs = 10, 50
  x = 0.001 * np.random.randn(num_inputs, num_classes)
  y = np.random.randint(num_classes, size=num_inputs)
  mode = 'cpu'
  
  mp_x = NumpyVarToMinpy(x)
  mp_y = NumpyVarToMinpy(y)
  dx_num = eval_numerical_gradient(lambda x: MinpyVarToNumpy(svm_loss(NumpyVarToMinpy(x), mp_y, mode)[0]), x, verbose=False)
  mp_loss, mp_dx = svm_loss(mp_x, mp_y, mode)
  
  dx = MinpyVarToNumpy(mp_dx)
  loss = MinpyVarToNumpy(mp_loss)

  # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
  print 'Testing svm_loss:'
  print 'loss: ', loss
  #print 'numerical error: ', dx_num
  #print 'analytical error: ', dx
  # Note: relative error would we large, because numeriacal error is unstable in gpu mode.
  print 'dx error: ', rel_error(dx_num, dx)
 

def Test_SVM_CPU_GPU():
  num_classes, num_inputs = 10, 50
  x = 0.001 * np.random.randn(num_inputs, num_classes)
  y = np.random.randint(num_classes, size=num_inputs)

  mp_x = NumpyVarToMinpy(x)
  mp_y = NumpyVarToMinpy(y) 
  mp_loss_cpu, mp_dx_cpu = svm_loss(mp_x, mp_y, 'cpu')
  mp_loss_gpu, mp_dx_gpu = svm_loss(mp_x, mp_y, 'gpu')

  loss_cpu = MinpyVarToNumpy(mp_loss_cpu)
  loss_gpu = MinpyVarToNumpy(mp_loss_gpu)
  dx_cpu = MinpyVarToNumpy(mp_dx_cpu)
  dx_gpu = MinpyVarToNumpy(mp_dx_gpu)

  # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
  print 'Testing svm_loss:'
  print 'cpu loss: ', loss_cpu 
  print 'gpu loss: ', loss_gpu 
  print 'dx error: ', rel_error(dx_cpu, dx_gpu)

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

  print ('Probability:', p)
  print ('Filterd ratio:', (out==0).sum()/100.0)
  
def Test_BatchNorm():
  # Check the training-time forward pass by checking means and variances
  # of features both before and after batch normalization
  
  # Simulate the forward pass for a two-layer network|
  N, D1, D2, D3 = 200, 50, 60, 3
  X = np.random.randn(N, D1)
  W1 = np.random.randn(D1, D2)
  W2 = np.random.randn(D2, D3)
  a = np.maximum(0, X.dot(W1)).dot(W2)
  
  print 'Before batch normalization:'
  print '  means: ', a.mean(axis=0)
  print '  stds: ', a.std(axis=0)
  
  # Means should be close to zero and stds close to one
  print 'After batch normalization (gamma=1, beta=0)'
  mp_a = NumpyVarToMinpy(a)
  mp_a_norm = batchnorm_forward(mp_a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
  a_norm = MinpyVarToNumpy(mp_a_norm)
  print '  mean: ', a_norm.mean(axis=0)
  print '  std: ', a_norm.std(axis=0)
  
  # Now means should be close to beta and stds close to gamma
  gamma = np.asarray([1.0, 2.0, 3.0])
  beta = np.asarray([11.0, 12.0, 13.0])
  mp_a_norm = batchnorm_forward(mp_a, gamma, beta, {'mode': 'train'})
  a_norm = MinpyVarToNumpy(mp_a_norm)
  print 'After batch normalization (nontrivial gamma, beta)'
  print '  means: ', a_norm.mean(axis=0)
  print '  stds: ', a_norm.std(axis=0)
  
#Test_Affine_Forward()
#Test_Affine_Backward()
#Relu_Forward()
#Relu_Backward()
#Test_SVM()
#Test_SVM_CPU_GPU()
#Test_Softmax()
#Test_Test_Forward()
#Test_Dropout()
Test_BatchNorm()
