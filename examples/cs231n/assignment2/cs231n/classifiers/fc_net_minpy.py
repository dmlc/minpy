"""
This file implements fullyconnected network in minpy.

All the array created in this file belongs to minpy.array Type.
Types of input values to loss() function, i.e. training/testing data & targets, should also be minpy.array.
"""
import numpy as py_np
import functools

from model import ModelBase
from cs231n.layers import affine_forward, relu_forward, svm_loss, dropout_forward, batchnorm_forward
from cs231n.layer_utils import affine_relu_forward

import minpy
import minpy.numpy as np
import minpy.numpy.random as random
from minpy.core import grad_and_loss

#import minpy.dispatch.policy as policy

class TwoLayerNet(ModelBase):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0, conv_mode='lazy', dtype=py_np.float64):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    super(TwoLayerNet, self).__init__(conv_mode)
    self.params = {}
    self.reg = reg

    self.params['W1'] = random.randn(input_dim, hidden_dim) * weight_scale 
    self.params['b1'] = np.zeros((hidden_dim))
    self.params['W2'] = random.randn(hidden_dim, num_classes) * weight_scale 
    self.params['b2'] = np.zeros((num_classes))

  def loss_and_derivative(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    # Note: types of X, y are mxnet.ndarray
    def train_loss(X, y, W1, W2, b1, b2):
      l1 = affine_relu_forward(X, W1, b1)
      l2 = affine_forward(l1, W2, b2)
      scores = l2

      if y is None:
        return scores
   
      #[TODO]: softmax is not supported yet
      # loss, d_scores = softmax_loss(scores, y)
      loss = svm_loss(scores, y)
      loss_with_reg = loss + np.sum(W1 ** 2) * 0.5 * self.reg + np.sum(W2 ** 2) * 0.5 * self.reg

      return loss_with_reg 

    self.params_array = []
    params_list_name = ['W1', 'W2', 'b1', 'b2']
    for param_name in params_list_name:
      self.params_array.append(self.params[param_name])

    X_plain = np.reshape(X, (X.shape[0], -1))
    if y is None:
      return train_loss(X_plain, y, *self.params_array)

    grad_function = grad_and_loss(train_loss, range(2, 6))

    grads_array, loss = grad_function(X_plain, y, *self.params_array)

    grads = {}
    for i in range(len(params_list_name)):
      grads[params_list_name[i]] = grads_array[i]

    return loss, grads


class FullyConnectedNet(ModelBase):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, seed=None, dtype=py_np.float64, conv_mode='lazy'):

    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    super(FullyConnectedNet, self).__init__(conv_mode)
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.params = {}

    #Define parameter name given # layer
    self.w_name = lambda l: 'W' + str(l)
    self.b_name = lambda l: 'b' + str(l)
    self.bn_ga_name = lambda l: 'bn_ga' + str(l)
    self.bn_bt_name = lambda l: 'bn_bt' + str(l)

    for l in range(self.num_layers):
      if l == 0:
        input_d = input_dim
      else:
        input_d = hidden_dims[l-1]

      if l < self.num_layers - 1:
        out_d = hidden_dims[l]
      else:
        out_d = num_classes

      self.params[self.w_name(l)] = random.randn(input_d, out_d) * weight_scale
      self.params[self.b_name(l)] = np.zeros((out_d))
      if l < self.num_layers and self.use_batchnorm:
        self.params[self.bn_ga_name(l)] = np.ones((out_d))
        self.params[self.bn_bt_name(l)] = np.zeros((out_d))

    self.param_keys = self.params.keys()

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Build key's index in loss func's arglist
    self.key_args_index = {}
    for i, key in enumerate(self.param_keys):
      # data, targets would be the first two elments in arglist
      self.key_args_index[key] = self.data_target_cnt + i

    # Init Key to index in loss_function args
    self.w_idx = self.wrap_param_idx(self.w_name)
    self.b_idx = self.wrap_param_idx(self.b_name)
    self.bn_ga_idx = self.wrap_param_idx(self.bn_ga_name)
    self.bn_bt_idx = self.wrap_param_idx(self.bn_bt_name)

  def wrap_param_idx(self, f):
    @functools.wraps(f)
    def find_idx(key):
      return self.key_args_index[f(key)]
    return find_idx

  def pack_params(self):
    params_collect = []
    for key in self.param_keys:
      params_collect.append(self.params[key])
    return params_collect

  def loss_and_derivative(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """

    X_plain = np.reshape(X, (X.shape[0], -1))
    mode = 'test' if y is None else 'train'

    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    params_array = self.pack_params()

    def train_loss(*args):
      X = args[0]
      y = args[1]

      res = X
      for l in xrange(self.num_layers):
        prev_res = res
        res = affine_forward(prev_res, args[self.w_idx(l)], args[self.b_idx(l)])

        if l < (self.num_layers - 1):
          if self.use_batchnorm:
            res = batchnorm_forward(res, args[self.bn_ga_idx(l)],
                                    args[self.bn_bt_idx(l)], self.bn_params[l])
          res = relu_forward(res)
          if self.use_dropout:
            res = dropout_forward(res, self.dropout_param)

      scores = res

      if mode == 'test':
        return scores

      #loss, _ = softmax_loss(scores, y)
      loss = svm_loss(scores, y)
      return loss

    if y is None:
      return train_loss(X_plain, y, *params_array)

    grad_function = grad_and_loss(train_loss, range(self.data_target_cnt, self.data_target_cnt + len(params_array)))
    grads_array, loss = grad_function(X_plain, y, *params_array)

    grads = {}

    for i, grad in enumerate(grads_array):
      grads[self.param_keys[i]] = grad
    return loss, grads
