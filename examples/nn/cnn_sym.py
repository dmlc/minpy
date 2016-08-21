"""Convolution Neural Network example using only MXNet symbol."""
import sys
import argparse

import minpy
import minpy.numpy as np
from minpy.core import Function
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter
from examples.utils.data_utils import get_CIFAR10_data

import mxnet as mx

batch_size=128

class ConvolutionNet(ModelBase):
    def __init__(self,
                 input_size=3 * 32 * 32,
                 hidden_size=512,
                 num_classes=10):
        super(ConvolutionNet, self).__init__()
        # Define symbols that using convolution and max pooling to extract better features
        # from input image.
        net = mx.sym.Variable(name='X')
        net = mx.sym.Reshape(data=net,
                             shape=(batch_size, 3, 32, 32))
        net = mx.sym.Convolution(data=net,
                                 name='conv',
                                 kernel=(7, 7),
                                 num_filter=32)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.Pooling(data=net,
                             name='pool',
                             pool_type='max',
                             kernel=(2, 2),
                             stride=(2, 2))
        net = mx.sym.Flatten(data=net)
        net = mx.sym.FullyConnected(name='fc1', data=net, num_hidden=hidden_size)
        net = mx.sym.Activation(data=net, act_type='relu')
        # Create forward function and add parameters to this model.
        net = mx.sym.FullyConnected(name='fc2', data=net, num_hidden=num_classes)
        self.cnn = Function(net, input_shapes={'X': (batch_size, input_size)},
                            name='cnn')
        self.add_params(self.cnn.get_params())

    def forward(self, X):
        out = self.cnn(X=X, **self.params)
        return out

    def loss(self, predict, y):
        return layers.softmax_loss(predict, y)

def main(args):
    model = ConvolutionNet()
    data = get_CIFAR10_data(args.data_dir)
    # reshape all data to matrix
    data['X_train'] = data['X_train'].reshape([data['X_train'].shape[0], 3 * 32 * 32])
    data['X_val'] = data['X_val'].reshape([data['X_val'].shape[0], 3 * 32 * 32])
    data['X_test'] = data['X_test'].reshape([data['X_test'].shape[0], 3 * 32 * 32])

    train_dataiter = NDArrayIter(data['X_train'],
                         data['y_train'],
                         batch_size=batch_size,
                         shuffle=True)

    test_dataiter = NDArrayIter(data['X_test'],
                         data['y_test'],
                         batch_size=batch_size,
                         shuffle=False)

    solver = Solver(model,
                    train_dataiter,
                    test_dataiter,
                    num_epochs=10,
                    init_rule='xavier',
                    update_rule='sgd_momentum',
                    optim_config={
                        'learning_rate': 1e-3,
                        'momentum': 0.9
                    },
                    verbose=True,
                    print_every=20)
    solver.init()
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-layer perceptron example using minpy operators")
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Directory that contains cifar10 data')
    main(parser.parse_args())
