"""Simple convolutional neural network on CIFAR10."""
import sys
import argparse

import mxnet as mx
import minpy
import minpy.numpy as np
import minpy.nn.io
import minpy.core
import minpy.nn.model
import minpy.nn.solver
from minpy.nn import layers
from examples.utils.data_utils import get_CIFAR10_data

import minpy.context
minpy.context.set_context(minpy.context.gpu(0))

batch_size = 128
input_size = (3, 32, 32)
hidden_size = 512
num_classes = 10


class ConvolutionNet(minpy.nn.model.ModelBase):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
        # Define symbols that using convolution and max pooling to extract better features
        # from input image.
        net = mx.sym.Variable(name='X')
        net = mx.sym.Convolution(
            data=net, name='conv', kernel=(7, 7), num_filter=32)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.Pooling(
            data=net,
            name='pool',
            pool_type='max',
            kernel=(2, 2),
            stride=(2, 2))
        net = mx.sym.Flatten(data=net)
        net = mx.sym.FullyConnected(
            data=net, name='fc1', num_hidden=hidden_size)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.FullyConnected(
            data=net, name='fc2', num_hidden=num_classes)
        self.cnn = minpy.core.Function(
            net, input_shapes={'X': (batch_size, ) + input_size}, name='cnn')
        self.add_params(self.cnn.get_params())

    def forward(self, X, mode):
        out = self.cnn(X=X, **self.params)
        return out

    def loss(self, predict, y):
        return layers.softmax_loss(predict, y)


def main(args):
    # Create model.
    model = ConvolutionNet()
    # Create data iterators for training and testing sets.
    data = get_CIFAR10_data(args.data_dir)
    train_dataiter = minpy.nn.io.NDArrayIter(
        data=data['X_train'],
        label=data['y_train'],
        batch_size=batch_size,
        shuffle=True)
    test_dataiter = minpy.nn.io.NDArrayIter(
        data=data['X_test'],
        label=data['y_test'],
        batch_size=batch_size,
        shuffle=False)
    # Create solver.
    solver = minpy.nn.solver.Solver(
        model,
        train_dataiter,
        test_dataiter,
        num_epochs=10,
        init_rule='gaussian',
        init_config={'stdvar': 0.001},
        update_rule='sgd_momentum',
        optim_config={'learning_rate': 1e-3,
                      'momentum': 0.9},
        verbose=True,
        print_every=20)
    # Initialize model parameters.
    solver.init()
    # Train!
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    main(parser.parse_args())
