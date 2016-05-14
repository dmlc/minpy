import minpy as np
from model import ModelBase

import mxnet as mx


class ModelInputDimInconsistencyError(ValueError):
    pass

class ThreeLayerConvNet(ModelBase): """
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

    self.params = {}
    self.reg = reg

    self.input_dim = input_dim
    C,H,W = input_dim
    F = num_filters
    filter_height,filter_width = filter_size,filter_size

    self.params['W1'] = random.randn(F,C,filter_height,filter_width) * weights
    self.params['b1'] = np.zeros(F)

    conv_stride = 1
    conv_pad = (filter_size-1)/2

    Hc,Wc = 1+(H+2*conv_pad-filter_height)/conv_stride,1+(W+2*conv_pad-filter_width)/conv_stride

    pool_height,pool_width = 2,2
    pool_stride = 2

    Hp,Wp = (Hc - pool_height)/pool_stride+1, (Wc - pool_width)/pool_stride+1

    self.params['W2'] = random.randn(Hp*Wp*F, hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    data = mx.sym.Variable(name='x')
    conv1 = mx.symbol.Convolution(name='conv', data=data, kernel=(5,5), num_filter=5)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
    
    flatten = mx.symbol.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(name='fc1', data=flatten, num_hidden=hidden_dim)
    act1 = mx.sym.Activation(data=fc1, act_type='sigmoid')

    fc2 = mx.sym.FullyConnected(name='fc2', data=fc1, num_hidden=hidden_dim)
    act2 = mx.sym.Activation(data=fc2, act_type='sigmoid')

    #  self.input_dim
    batch_num, x_c, x_h, x_w = X.shape
    c, h, w = self.input_dim
    if not ( c == x_c and h == x_h and x_w ):
      raise ModelInputDimInconsistencyError('Expected Dim: {}, Input Dim: {}'.format(self.input_dim, X.shape))
    f = core.function(act2, [('x', xshape)])

    # TODO: Add loss function & parameter update settings
    scores = affine_forward(affine_out,W3,b3)

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dscores = softmax_loss(scores,y)

    loss += 0.5*self.reg*np.sum(W3**2)
    loss += 0.5*self.reg*np.sum(W2**2)
    loss += 0.5*self.reg*np.sum(W1**2)

    dx,dW3,db3 = affine_backward(dscores,scores_cache)
    dW3 += self.reg*W3
    grads['W3'] = dW3
    grads['b3'] = db3

    dx,dW2,db2 = affine_relu_backward(dx,affine_cache)
    dW2 += self.reg*W2
    grads['W2'] = dW2
    grads['b2'] = db2

    dx = dx.reshape((N,F,Hp,Wp))

    dx,dW1,db1 = conv_relu_pool_backward(dx,conv_cache)
    dW1 += self.reg*W1
    grads['W1'] = dW1
    grads['b1'] = db1


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
