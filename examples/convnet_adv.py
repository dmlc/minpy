"""Convolutional neural net on MNIST, modeled on 'LeNet-5',
http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf"""
import minpy
import minpy.numpy as np
import minpy.caffe as caffe
from builtins import range

def make_nn_funs(input_shape, L2_reg):

    def predictions(caffe_layer_specs, inputs):
        """Outputs normalized log-probabilities.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for caffe_layer in caffe_layer_specs:
            # original code:
            # cur_weights = parser.get(W_vect, caffe_layer)
            # cur_units = layer.forward_pass(cur_units, cur_weights)
            cur_units = caffe_layer.ff(cur_units)
        return cur_units

    def loss(caffe_layer_specs, X, T):
        # original code:
        # log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_prior = 0
        for caffe_layer in caffe_layer_specs:
            log_prior += -L2_reg * np.dot(caffe_layer.get_learnable_params()[0], caffe_layer.get_learnable_params()[0])

        log_lik = np.sum(predictions(caffe_layer_specs, X) * T)
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

    # how to specify input size && get output size in an elegant way? 
    caffe_layer_specs = [
        caffe.ConvolutionLayer(
            filter_window = (5, 5), 
            feature_maps = 6)),

        caffe.MaxPoolingLayer(
            pooling_window = (2, 2)),

        caffe.ConvolutionLayer(
            filter_window = (5, 5), 
            feature_maps = 16),

        caffe.MaxPoolingLayer(
            pooling_window = (2, 2)),

        caffe.FullyConnectedLayer(
            num_outputs = 120),

        caffe.TanhLayer(),

        caffe.FullyConnectedLayer(
            num_outputs = 84),

        caffe.TanhLayer(),

        caffe.FullyConnectedLayer(
            num_outputs = 10),

        caffe.SoftMaxLayer()]

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, L2_reg)
    loss_grad = grad(loss_fun)
