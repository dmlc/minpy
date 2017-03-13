""" DNN Layers. """
from __future__ import division

import minpy.numpy as np
import minpy.numpy.random as random

# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, no-member


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
    # TODO: fix NDArray type system
    N, D = x.shape
    N, D = int(N), int(D)
    if running_mean is None:
        running_mean = np.zeros(D)
    if running_var is None:
        running_var = np.zeros(D)

    out = None
    if mode == 'train':
        mean = np.sum(x, axis=0) / N
        x_mean = (x - np.expand_dims(mean, axis=0))
        sqr_x_mean = x_mean**2
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
        out = x * (1 - prob)
    return out


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


def softmax_cross_entropy(prob, label):
    """
    Computes the cross entropy for softmax activation.

    Inputs:
    - prob: Probability, of shape (N, C) where x[i, j] is the probability for the jth class
      for the ith input.
    - label: Either of the followings:
      - One hot encoding of labels, of shape (N, C)
      - Label index of shape (N, ), each y[i] is the label of i^th example
        (0 <= y[i] < C)

    Returns a Value:
    - cross_entropy
    """

    N = prob.shape[0]
    C = prob.shape[1]
    if len(label.shape) == 1:
        #convert it to one hot encoding
        onehot_label = np.zeros([N, C])
        np.onehot_encode(label, onehot_label)
    else:
        onehot_label = label
    return -np.sum(np.log(prob) * onehot_label) / N


def softmax_loss(x, label):
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
    if len(label.shape) == 1:
        #convert it to one hot encoding
        onehot_label = np.zeros([N, C])
        np.onehot_encode(label, onehot_label)
    else:
        onehot_label = label
    prob = np.softmax_output(x, onehot_label)
    return softmax_cross_entropy(prob, onehot_label)


def l2_loss(x, label):
    """
    The Mean Square Error loss for regression.
    """
    N = x.shape[0]
    C = x.shape[1]
    if len(label.shape) == 1:
        #convert it to one hot encoding
        onehot_label = np.zeros([N, C])
        np.onehot_encode(label, onehot_label)
    else:
        onehot_label = label
    return np.sum((x - onehot_label)**2) / N


def sigmoid(x):
    """
    Computes the forward pass for a layer of sigmoid units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    """

    return 1 / (1 + np.exp(-x))


def rnn_step(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    """
    next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
    return next_h


def rnn_temporal(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    """
    N, T, _ = x.shape
    H = h0.shape[1]
    h = np.zeros([N, 0, H])
    for t in range(T):
        h_step = rnn_step(x[:, t, :], h0 if t == 0 else h[:, t - 1, :], Wx, Wh,
                          b).reshape(N, 1, H)
        h = np.append(h, h_step, axis=1)
    return h


def gru_step(x, prev_h, Wx, Wh, b, Wxh, Whh, bh):
    """
    Forward pass for a single timestep of an GRU.

    The input data has dimentsion D, the hidden state has dimension H, and we
    use a minibatch size of N.

    Parameters
    ----------
    x
        Input data, of shape (N, D)
    prev_h
        Previous hidden state, of shape (N, H)
    prev_c
        Previous hidden state, of shape (N, H)
    Wx
        Input-to-hidden weights for r and z gates, of shape (D, 2H)
    Wh
        Hidden-to-hidden weights for r and z gates, of shape (H, 2H)
    b
        Biases for r an z gates, of shape (2H,)
    Wxh
        Input-to-hidden weights for h', of shape (D, H)
    Whh
        Hidden-to-hidden weights for h', of shape (H, H)
    bh
        Biases, of shape (H,)

    Returns
    -------
    next_h
        Next hidden state, of shape (N, H)

    Notes
    -----
    Implementation follows
    http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    """
    _, H = prev_h.shape
    a = sigmoid(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
    r = a[:, 0:H]
    z = a[:, H:2 * H]
    h_m = np.tanh(np.dot(x, Wxh) + np.dot(r * prev_h, Whh) + bh)
    next_h = z * prev_h + (1 - z) * h_m
    return next_h


def lstm_step(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    """
    _, H = prev_c.shape
    # 1. activation vector
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    # 2. gate fuctions
    i = sigmoid(a[:, 0:H])
    f = sigmoid(a[:, H:2 * H])
    o = sigmoid(a[:, 2 * H:3 * H])
    g = np.tanh(a[:, 3 * H:4 * H])
    # 3. next cell state
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)
    return next_h, next_c


def lstm_temporal(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """
    N, T, _ = x.shape
    _, H = h0.shape
    c = np.zeros([N, 0, H])
    h = np.zeros([N, 0, H])
    for t in range(T):
        h_step, c_step = lstm_step(x[:, t, :], h[:, t - 1, :]
                                   if t > 0 else h0, c[:, t - 1, :]
                                   if t > 0 else np.zeros((N, H)), Wx, Wh,
                                   b)  # pylint: disable=line-too-long
        h_step = h_step.reshape(N, 1, H)
        c_step = c_step.reshape(N, 1, H)
        h = np.append(h, h_step, axis=1)
        c = np.append(c, c_step, axis=1)
    return h


def temporal_affine(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N

    return loss
