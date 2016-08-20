import minpy
import minpy.numpy as np
import minpy.numpy.random as random
from minpy.core import wraps
from minpy.array_variants import ArrayType
import minpy.dispatch.policy as policy

np.set_policy(policy.OnlyNumPyPolicy())
random.set_policy(policy.OnlyNumPyPolicy())

@wraps('lazy')
def affine_forward(x, w, b):
    """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """

    out = np.dot(x, w) + b
    cache = (x, w, b)

    return out, cache

@wraps('lazy')
def affine_backward(dout, cache):
    """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
    x, w, b = cache
    x_plain = np.reshape(x, (x.shape[0], -1))

    db = np.sum(dout, axis=0)

    dx_plain = np.dot(dout, np.transpose(w))

    dx = np.reshape(dx_plain, x.shape)
    dw = np.dot(np.transpose(x_plain), dout)

    return dx, dw, db

@wraps('lazy')
def relu_forward(x):
    """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
    out = np.maximum(0, x)
    cache = x
    return out, cache

@wraps('lazy')
def relu_backward(dout, cache):
    """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################

    x = cache
    dx = dout * (x > 0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


@wraps('lazy')
def batchnorm_forward(x, gamma, beta, running_mean, running_var, bn_param):
    """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    if running_mean is None:
        running_mean = np.zeros(D)
    if running_var is None:
        running_var = np.zeros(D)

    out = None
    cache = None
    if mode == 'train':
        mean = np.sum(x, axis=0) / N
        x_mean = (x - mean)
        sqr_x_mean = x_mean**2
        var = np.sum(sqr_x_mean, axis=0) / N
        sqrt_var = np.sqrt(var + eps)

        inv_sqrt_var = 1.0 / sqrt_var

        x_hat = x_mean * inv_sqrt_var
        out = gamma * x_hat + beta

        cache = (x_hat,gamma, sqr_x_mean, mean, var, sqrt_var, x_mean, inv_sqrt_var)
        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * var
    elif mode == 'test':
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, running_mean, running_var, cache

@wraps('lazy')
def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  x_hat, gamma, sqr_x_mean, mean, var, sqrt_var, x_mean, inv_sqrt_var = cache
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  N = x_hat.shape[0]

  dx_hat = dout * gamma
  dx_mean = dx_hat*inv_sqrt_var
  dinv_sqrt_var = np.sum(dx_hat*x_mean,axis=0)
  dsqrt_var = -1.0/(sqrt_var**2)*dinv_sqrt_var
  dvar = 0.5*inv_sqrt_var*dsqrt_var
  dsqr_x_mean = 1.0/N*np.ones(sqr_x_mean.shape)*dvar
  dx_mean += 2*x_mean*dsqr_x_mean

  dmean = -np.sum(dx_mean,axis=0)

  dx1 = dx_mean
  dx2 = 1.0/N*np.ones(mean.shape)*dmean

  dx = dx1+dx2
  dgamma = np.sum(x_hat*dout,axis=0)
  dbeta =  np.sum(dout,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta

@wraps('lazy')
def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.

  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.

  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.

  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  x_hat,gamma,sqr_x_mean,mean,var,sqrt_var,x_mean,inv_sqrt_var = cache
  N = x_hat.shape[0]
  dbeta = np.sum(dout,axis=0)
  dgamma = np.sum(dout*x_hat,axis=0)

  dx = 1.0/N*inv_sqrt_var*gamma*(
   N*dout
   -np.sum(dout,axis=0)
   -x_mean*(inv_sqrt_var**2)*np.sum(dout*x_mean,axis=0)
  )
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


@wraps('lazy')
def dropout_forward(x, dropout_param):
    """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #TODO: check implementation of compare operator in mxnet?
        mask = random.rand(*x.shape) > p
        out = x * mask  #drop!
    else:
        out = x

    cache = (dropout_param, mask)
    return out, cache

@wraps('lazy')
def dropout_backward(dout, cache):
    """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        dx = dout*mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def max_pool_forward_naive(x, pool_param):
    """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


@wraps('lazy')
def svm_loss_forward(x, y):
    """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - cache: (x, y)
  """

    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]

    #TODO: Support broadcast case: (X,) (X, Y)
    #shape(x) is (d0, d1)
    #shape(correct_class_scores) is (d0,)
    #margins = np.maximum(0, x - correct_class_scores + 1.0)
    margins = np.transpose(np.maximum(0, np.transpose(x) - np.transpose(
        correct_class_scores) + 1.0))
    cache = (x, y)

    loss = (np.sum(margins) - np.sum(margins[np.arange(N), y])) / N
    return loss, cache

@wraps('lazy')
def svm_loss_backward(cache):
    """
  Computes the backward pass for a svm loss function.

  Inputs:
  - cache:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, C)
  """
    x, y = cache

    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.transpose(np.maximum(0, np.transpose(x) - np.transpose(
        correct_class_scores) + 1.0))
    margins[np.arange(N), y] = 0
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N

    return dx


@wraps('lazy')
def softmax_loss_forward(x, y):
    """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - cache: (x, y)
  """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    cache = (x, y)
    return loss, cache

@wraps('lazy')
def softmax_loss_backward(cache):
    """
  Computes the backward pass for a softmax loss function.

  Inputs:
  - cache:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, C)
  """
    x, y = cache
    N = x.shape[0]

    probs = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    dx = probs / np.expand_dims(np.sum(probs, axis=1), axis=1)
    dx[np.arange(N), y] -= 1
    dx /= N
    return dx
