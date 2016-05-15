import minpy 
import minpy.numpy as np
import mxnet as mx
import minpy.numpy.random as random
from model import ModelBase
import minpy.core as core

class ModelInputDimInconsistencyError(ValueError):
  pass

class ThreeLayerConvNet(ModelBase): 
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
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
    # this should be moved in super() init
    self.symbol_func = None

    self.input_dim = input_dim
    self.num_filters = num_filters 
    self.filter_size = filter_size
    self.hidden_dim = hidden_dim 
    self.reg = reg
    self.num_classes = num_classes
    self.weight_scale = weight_scale

    self.set_param()

  def set_param(self):
    self.params = {}

    c_cnt, height, width = self.input_dim
    f_cnt = self.num_filters
    f_h, f_w = self.filter_size, self.filter_size

    self.params['conv_weight'] = random.randn(f_cnt, c_cnt, f_h, f_w) * self.weight_scale
    self.params['conv_bias'] = np.zeros(f_cnt)

    conv_stride = 1
    conv_pad = (f_h-1) / 2

    Hc, Wc = 1 + ( height + 2*conv_pad - f_h) / conv_stride , 1 + ( width + 2*conv_pad - f_w) / conv_stride

    pool_height, pool_width = 2, 2
    pool_stride = 2

    Hp, Wp = (Hc - pool_height)/pool_stride+1, (Wc - pool_width)/pool_stride+1

    self.params['fc1_weight'] = random.randn(Hp*Wp*f_cnt, self.hidden_dim) * self.weight_scale
    self.params['fc1_bias'] = np.zeros(self.hidden_dim)

    self.params['fc2_weight'] = random.randn(self.hidden_dim, self.num_classes) * self.weight_scale
    self.params['fc2_bias'] = np.zeros(self.num_classes)

    self.param_keys = self.params.keys()

    # Build key's index in loss func's arglist
    self.key_args_index = {}
    for i, key in enumerate(self.param_keys):
      # data, targets would be the first two elments in arglist
      self.key_args_index[key] = self.data_target_cnt + i

  def set_mxnet_symbol(self, X):

    data = mx.sym.Variable(name='x')
    conv1 = mx.symbol.Convolution(name='conv', data=data, kernel=(5,5), num_filter=5)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
    
    flatten = mx.symbol.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(name='fc1', data=flatten, num_hidden=self.hidden_dim)
    act1 = mx.sym.Activation(data=fc1, act_type='sigmoid')

    fc2 = mx.sym.FullyConnected(name='fc2', data=fc1, num_hidden=self.hidden_dim)
    act2 = mx.sym.Activation(data=fc2, act_type='sigmoid')

    batch_num, x_c, x_h, x_w = X.shape
    c, h, w = self.input_dim
    if not ( c == x_c and h == x_h and x_w ):
      raise ModelInputDimInconsistencyError('Expected Dim: {}, Input Dim: {}'.format(self.input_dim, X.shape))
    self.symbol_func = core.function(act2, [('x', X.shape)])


  def pack_params(self):
    params_collect = []
    for key in self.param_keys:
      params_collect.append(self.params[key])
    return params_collect

  def get_index_reg_weight(self):
    return [self.key_args_index[key] for key in ['conv_weight', 'fc1_weight', 'fc2_weight']]

  def make_mxnet_weight_dict(self, args):
    wDict = {}
    assert len(args) == len(self.param_keys)
    for i, key in enumerate(self.param_keys):
      wDict[key] = args[i]
    wDict['x'] = args[0]
    return wDict

  def loss_and_derivative(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    if self.symbol_func == None:
      self.set_mxnet_symbol(X)

    params_array = self.pack_params()

    def train_loss(*args):
      inputs = args[0]
      targets = args[1]
      probs = self.symbol_func(**self.make_mxnet_weight_dict(args[self.data_target_cnt:len(args)]))

      if targets is None:
        return probs 

      N = inputs.shape[0]
      loss = -np.sum(np.log(probs[np.arange(N), targets])) / N

      for reg_index in self.get_index_reg_weight():
        loss += 0.5*self.reg*np.sum(args[reg_index]**2)
      return loss

    if y is None:
      return train_loss(X, y, *params_array)

    grad_function = core.grad_and_loss(train_loss, range(self.data_target_cnt, self.data_target_cnt + len(params_array)))
    grads_array, loss = grad_function(X, y, *params_array)

    grads = {}

    for i, grad in enumerate(grads_array):
      grads[self.param_keys[i]] = grad

    return loss, grads
