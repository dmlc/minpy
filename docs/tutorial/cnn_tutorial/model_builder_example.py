'''
  This example demonstrates how to use minpy model builder to construct neural networks.

  For details about how to train a model with solver, please refer to:
    http://minpy.readthedocs.io/en/latest/tutorial/complete.html

  More models are available in minpy.nn.model_gallery.
'''

import sys
import argparse

import minpy.nn.model_builder as builder
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter
from examples.utils.data_utils import get_CIFAR10_data

# Please uncomment following if you have GPU-enabled MXNet installed.
#from minpy.context import set_context, gpu
#set_context(gpu(0)) # set the global context as gpu(0)

batch_size = 128
hidden_size = 512
num_classes = 10

def main(args):
    # Define a convolutional neural network the same as above
    net = builder.Sequential(
        builder.Convolution((7, 7), 32),
        builder.ReLU(),
        builder.Pooling('max', (2, 2), (2, 2)),
        builder.Flatten(),
        builder.Affine(hidden_size),
        builder.Affine(num_classes),
    )

    # Cast the definition to a model compatible with minpy solver
    model = builder.Model(net, 'softmax', (3 * 32 * 32,))

    data = get_CIFAR10_data(args.data_dir)

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
    solver.init()
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-layer perceptron example using minpy operators")
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Directory that contains cifar10 data')
    main(parser.parse_args())
