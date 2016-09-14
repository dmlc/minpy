'''
  This example demonstrates how to use minpy model builder to construct neural networks.

  For details about how to train a model with solver, please refer to:
    http://minpy.readthedocs.io/en/latest/tutorial/complete.html

  More models are available in minpy.nn.model_gallery.
'''

import sys
import argparse

import minpy
import minpy.numpy as np
from minpy.nn import layers
from minpy.nn.model import ModelBase
import minpy.nn.model_builder as builder
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter
from examples.utils.data_utils import get_CIFAR10_data

# specify the GPU on which minpy performs computation
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    # Define a 2-layer perceptron
    MLP = builder.Sequential(
        builder.Affine(512),
        builder.ReLU(),
        builder.Affine(10)
    )

    # Cast the definition to a model compatible with minpy solver
    model = builder.Model(MLP, 'softmax', (3 * 32 * 32,))

    data = get_CIFAR10_data(args.data_dir)
    data['X_train'] = data['X_train'].reshape([data['X_train'].shape[0], 3 * 32 * 32])
    data['X_val'] = data['X_val'].reshape([data['X_val'].shape[0], 3 * 32 * 32])
    data['X_test'] = data['X_test'].reshape([data['X_test'].shape[0], 3 * 32 * 32])

    train_dataiter = NDArrayIter(data['X_train'],
                         data['y_train'],
                         batch_size=100,
                         shuffle=True)

    test_dataiter = NDArrayIter(data['X_test'],
                         data['y_test'],
                         batch_size=100,
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
                        'learning_rate': 1e-5,
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
