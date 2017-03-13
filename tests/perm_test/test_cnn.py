"""Convolution Neural Network example with both MinPy ndarray and MXNet symbol."""
import sys
import argparse

import minpy
import minpy.numpy as np
import mxnet as mx
from minpy.nn.io import NDArrayIter
# Can also use MXNet IO here
# from mxnet.io import NDArrayIter
from minpy.core import Function
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from examples.utils.data_utils import get_CIFAR10_data

# Please uncomment following if you have GPU-enabled MXNet installed.
#from minpy.context import set_context, gpu
#set_context(gpu(0)) # set the global context as gpu(0)

def test_cnn():
    batch_size=128
    input_size=(3, 32, 32)
    flattened_input_size=3 * 32 * 32
    hidden_size=512
    num_classes=10
    
    class ConvolutionNet(ModelBase):
        def __init__(self):
            super(ConvolutionNet, self).__init__()
            # Define symbols that using convolution and max pooling to extract better features
            # from input image.
            net = mx.sym.Variable(name='X')
            net = mx.sym.Convolution(
                    data=net, name='conv', kernel=(7, 7), num_filter=32)
            net = mx.sym.Activation(
                    data=net, act_type='relu')
            net = mx.sym.Pooling(
                    data=net, name='pool', pool_type='max', kernel=(2, 2),
                    stride=(2, 2))
            net = mx.sym.Flatten(data=net)
            # Create forward function and add parameters to this model.
            self.conv = Function(
                    net, input_shapes={'X': (batch_size,) + input_size},
                    name='conv')
            self.add_params(self.conv.get_params())
            # Define ndarray parameters used for classification part.
            output_shape = self.conv.get_one_output_shape()
            conv_out_size = output_shape[1]
            self.add_param(name='w1', shape=(conv_out_size, hidden_size)) \
                .add_param(name='b1', shape=(hidden_size,)) \
                .add_param(name='w2', shape=(hidden_size, num_classes)) \
                .add_param(name='b2', shape=(num_classes,))
    
        def forward(self, X, mode):
            out = self.conv(X=X, **self.params)
            out = layers.affine(out, self.params['w1'], self.params['b1'])
            out = layers.relu(out)
            out = layers.affine(out, self.params['w2'], self.params['b2'])
            # This verifies whether symbols can be reused.
            trash = self.conv(X=np.zeros(X.shape), **self.params)
            return out
    
        def loss(self, predict, y):
            return layers.softmax_loss(predict, y)
    
    def main():
        # Create model.
        model = ConvolutionNet()
        # Create data iterators for training and testing sets.
        data = get_CIFAR10_data('cifar-10-batches-py')
        train_dataiter = NDArrayIter(data=data['X_train'],
                                     label=data['y_train'],
                                     batch_size=batch_size,
                                     shuffle=True)
        test_dataiter = NDArrayIter(data=data['X_test'],
                                    label=data['y_test'],
                                    batch_size=batch_size,
                                    shuffle=False)
        # Create solver.
        solver = Solver(model,
                        train_dataiter,
                        test_dataiter,
                        num_epochs=1,
                        init_rule='gaussian',
                        init_config={
                            'stdvar': 0.001
                        },
                        update_rule='sgd_momentum',
                        optim_config={
                            'learning_rate': 1e-3,
                            'momentum': 0.9
                        },
                        verbose=True,
                        print_every=20)
        # Initialize model parameters.
        solver.init()
        # Train!
        solver.train()

        train_acc = solver.check_accuracy(
            train_dataiter, num_samples=solver.train_acc_num_samples)

        # a normal cnn should reach 50% train acc
        assert (train_acc >= 0.40)

    """ Have trouble using argparse in nosetest
    parser = argparse.ArgumentParser(description="Multi-layer perceptron example using minpy operators")
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../examples/dataset/cifar10/cifar-10-batches-py')
    """
    main()

if __name__ == '__main__':
    test_cnn()
