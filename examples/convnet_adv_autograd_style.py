"""Convolutional neural net on MNIST, modeled on 'LeNet-5',
http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf"""
from __future__ import absolute_import
from __future__ import print_function

import minpy
import minpy.numpy as np
import minpy.caffe as caffe
from builtins import range

class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))
        start += N_batch
    return batches

def logsumexp(X, axis, keepdims=False):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=keepdims))

def make_nn_funs(input_shape, caffe_layer_specs, L2_reg):
    parser = WeightsParser()
    cur_shape = input_shape
    for caffe_layer in caffe_layer_specs:
        # oroginal code :
        # N_weights, cur_shape = layer.build_weights_dict(cur_shape)
        N_weights, cur_shape = caffe_layer.BuildShape(cur_shape)
        parser.add_weights(caffe_layer, (N_weights,))

    def predictions(W_vect, inputs):
        """Outputs normalized log-probabilities.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for caffe_layer in caffe_layer_specs:
            cur_weights = parser.get(W_vect, caffe_layer)
            # original code:
            # cur_units = layer.forward_pass(cur_units, cur_weights)
            cur_units = caffe_layer.ff(cur_units, cur_weights)
        return cur_units

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T)
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(pred_fun(W_vect, X), axis=1))

    return parser.N, predictions, loss, frac_err

if __name__ == '__main__':
    # Network parameters
    L2_reg = 1.0
    input_shape = (1, 28, 28)
    '''
    original code:
    layer_specs = [conv_layer((5, 5), 6),
                   maxpool_layer((2, 2)),
                   conv_layer((5, 5), 16),
                   maxpool_layer((2, 2)),
                   tanh_layer(120),
                   tanh_layer(84),
                   softmax_layer(10)]
    '''

    caffe_layer_specs = [caffe.ConvolutionLayer((5, 5), 6),
                   caffe.MaxPoolingLayer((2, 2)),
                   caffe.ConvolutionLayer((5, 5), 16),
                   caffe.MaxPoolingLayer((2, 2)),
                   caffe.FullyConnectedLayer(120),
                   caffe.TanhLayer(),
                   caffe.FullyConnectedLayer(84),
                   caffe.TanhLayer(),
                   caffe.FullyConnectedLayer(10),
                   caffe.SoftMaxLayer()]

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 256
    num_epochs = 50

    # Load and process MNIST data (borrowing from Kayak)
    print("Loading training data...")
    import imp, urllib
    add_color_channel = lambda x : x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    source, _ = urllib.urlretrieve(
        'https://raw.githubusercontent.com/HIPS/Kayak/master/examples/data.py')
    data = imp.load_source('data', source).mnist()
    train_images, train_labels, test_images, test_labels = data
    train_images = add_color_channel(train_images) / 255.0
    test_images  = add_color_channel(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, caffe_layer_specs, L2_reg)
    loss_grad = grad(loss_fun)

    # Initialize weights
    rs = npr.RandomState()
    W = rs.randn(N_weights) * param_scale

    # Check the gradients numerically, just to be safe
    # quick_grad_check(loss_fun, W, (train_images[:50], train_labels[:50]))

    print("    Epoch      |    Train err  |   Test error  ")
    def print_perf(epoch, W):
        test_perf  = frac_err(W, test_images, test_labels)
        train_perf = frac_err(W, train_images, train_labels)
        print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    # Train with sgd
    batch_idxs = make_batches(N_data, batch_size)
    cur_dir = np.zeros(N_weights)

    for epoch in range(num_epochs):
        print_perf(epoch, W)
        for idxs in batch_idxs:
            grad_W = loss_grad(W, train_images[idxs], train_labels[idxs])
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
            W -= learning_rate * cur_dir
