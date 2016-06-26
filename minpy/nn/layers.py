import minpy
import minpy.numpy as np
import minpy.numpy.random as random
from minpy.core import wraps

@wraps('lazy')
def affine(x, w, b):
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
    """
    out = np.dot(x, w) + b
    return out

@wraps('lazy')
def relu(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    """
    out = np.maximum(0, x)
    return out


@wraps('lazy')
def batchnorm(x,
              gamma,
              beta,
              mode='train',
              eps=1e-5,
              momentum=0.9,
              running_mean=None,
              running_var=None):
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
    - mode: 'train' or 'test'
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - running_mean: updated running_mean
    - running_var: updated running_var
  """
    N, D = x.shape
    if running_mean is None:
        running_mean = np.zeros(D)
    if running_var is None:
        running_var = np.zeros(D)

    out = None
    if mode == 'train':
        mean = np.sum(x, axis=0) / N
        x_mean = (x - np.expand_dims(mean, axis=0))
        sqr_x_mean = x_mean ** 2
        var = np.sum(sqr_x_mean, axis=0) / N
        sqrt_var = np.sqrt(var + eps)
        inv_sqrt_var = 1.0 / sqrt_var
        x_hat = x_mean * np.expand_dims(inv_sqrt_var, axis=0)
        out = gamma * x_hat + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * var
    elif mode == 'test':
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # return the updated running means 
    return out, running_mean, running_var

@wraps('lazy')
def dropout(x, prob, mode='train', seed=None):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - prob: Dropout parameter. We drop each neuron output with probability prob.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

    Outputs:
    - out: Array of the same shape as x.
    """
    if seed is not None:
        random.seed(seed)
    mask = None
    out = None
    if mode == 'train':
        #TODO: check implementation of compare operator in mxnet?
        mask = random.rand(*x.shape) > prob
        out = x * mask  #drop!
    else:
        out = x
    return out


@wraps('lazy')
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    #TODO: Support broadcast case: (X,) (X, Y)
    #shape(x) is (d0, d1)
    #shape(correct_class_scores) is (d0,)
    #margins = np.maximum(0, x - correct_class_scores + 1.0)
    margins = np.maximum(0, x - correct_class_scores + 1.0)
    loss = (np.sum(margins) - np.sum(margins[np.arange(N), y])) / N
    return loss

@wraps('lazy')
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Either of the followings:
      - One hot encoding of labels, of shape (N, C)
      - Label index of shape (N, ), each y[i] is the label of i^th example
        (0 <= y[i] < C)

    Returns a tuple of:
    - loss: Scalar giving the loss
    """
    N = x.shape[0]
    C = x.shape[1]
    if len(y.shape) == 1:
        #convert it to one hot encoding
        onehot_y = np.zeros([N, C])
        np.onehot_encode(y, onehot_y)
    else:
        onehot_y = y
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(np.log(probs) * onehot_y) / N
    return loss

@wraps('lazy')
def l2_loss(x, y):
    return np.sum((x - y) ** 2)
