import minpy
import minpy.numpy as np
import minpy.numpy.random as random
import minpy.core as core

import mxnet as mx
import numpy as py_np

from model import ModelBase
from cs231n.layers import  softmax_loss_forward


class ThreeLayerConvNet(ModelBase):
    """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

    def __init__(self,
                 input_dim=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=py_np.float64,
                 conv_mode='lazy'):
        """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    """
        super(ThreeLayerConvNet, self).__init__(conv_mode)
        #TODO(haoran): Add another parent class
        #This should be moved into super() __init__
        self.symbol_func = None
        self.params = {}

        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.reg = reg
        self.num_classes = num_classes
        self.weight_scale = weight_scale

    def set_param(self):
        self.params = {}
        executor = self.symbol_func._executor
        for k, v in executor.arg_dict.items():
            if (k != "X"):
                self.params[k] = random.rand(*v.shape) * self.weight_scale

        #TODO(Haoran): move following into parent structured model class
        self.param_keys = self.params.keys()

        # Build key's index in loss func's arglist
        self.key_args_index = {}
        for i, key in enumerate(self.param_keys):
            # data, targets would be the first two elments in arglist
            self.key_args_index[key] = self.data_target_cnt + i

    def set_mxnet_symbol(self, X):

        data = mx.sym.Variable(name='X')
        conv1 = mx.symbol.Convolution(name='conv1',
                                      data=data,
                                      kernel=(self.filter_size,
                                              self.filter_size),
                                      num_filter=self.num_filters)
        tanh1 = mx.symbol.Activation(data=conv1, act_type="relu")
        pool1 = mx.symbol.Pooling(data=tanh1,
                                  pool_type="max",
                                  kernel=(2, 2),
                                  stride=(2, 2))

        fc1 = mx.sym.FullyConnected(name='fc1',
                                    data=pool1,
                                    num_hidden=self.hidden_dim)

        act1 = mx.sym.Activation(data=fc1, act_type='relu')

        fc2 = mx.sym.FullyConnected(name='fc2',
                                    data=fc1,
                                    num_hidden=self.num_classes)

        self.symbol_func = core.Function(
                fc2, {'X': X.shape})


    def loss_and_derivative(self, X, y=None):
        # symbol's init func takes input size.
        if self.symbol_func == None:
            self.set_mxnet_symbol(X)
            self.set_param()

        def train_loss(*args):
            probs = self.symbol_func(X=X,
                                     conv1_weight=self.params['conv1_weight'],
                                     conv1_bias=self.params['conv1_bias'],
                                     fc1_weight=self.params['fc1_weight'],
                                     fc1_bias=self.params['fc1_bias'],
                                     fc2_weight=self.params['fc2_weight'],
                                     fc2_bias=self.params['fc2_bias'])
            if y is None:
                return probs

            loss, _ = softmax_loss_forward(probs, y)
            loss = loss + 0.5 * self.reg *\
                    (np.sum(self.params['conv1_weight'])+
                     np.sum(self.params['fc1_weight'])+
                     np.sum(self.params['fc2_weight']))
            return loss

        if y is None:
            return train_loss()

        param_keys = list(self.params.keys())
        param_arrays = list(self.params.values())
        grad_function = core.grad_and_loss(train_loss, range(len(param_arrays)))
        grad_arrays, loss = grad_function(*param_arrays)

        grads = dict(zip(param_keys, grad_arrays))
        return loss, grads
