""" Convolutional Neural Network Implemented by Minpy.model_builder """

from minpy.nn.solver import Solver
import minpy.nn.model_builder as builder

def main():
    # Specify network configurations.
    CNN = builder.Sequential(
        builder.Convolution((7, 7), 32),
        builder.ReLU(),
        builder.Pooling('max', (2, 2), (2, 2)),
        builder.Flatten(),
        builder.Affine(512),
        builder.ReLU(),
        builder.Affine(10)
    )

    # Cast the configuration to a model compatible with minpy solver.
    model = builder.Model(CNN, 'softmax', (3, 32, 32))

    # The above line is equivalent to using the ConvolutionNet, as below(uncomment).
    # model = TwoLayerNet()

    '''
    solver = Solver(
        model,
        # ...
    )
    '''

    # Training procedures are identical to previous examples.

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
from minpy.context import set_context, cpu
set_context(cpu())
def main(args):
    CNN = builder.Sequential(
        builder.Convolution((7, 7), 32),
        builder.ReLU(),
        builder.Pooling('max', (2, 2), (2, 2)),
        builder.Flatten(),
        builder.Affine(512),
        builder.ReLU(),
        builder.Affine(10)
    )


    # Create model.
    model = builder.Model(CNN, 'softmax', (3, 32, 32))
    # Create data iterators for training and testing sets.
    data = get_CIFAR10_data(args.data_dir)
    print 'here'
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
                    num_epochs=10,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-layer perceptron example using minpy operators")
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Directory that contains cifar10 data')
    main(parser.parse_args())
