""" Simple multi-layer perception neural network using Minpy """
from __future__ import print_function

import sys
import argparse
import minpy
from examples.utils.data_utils import get_CIFAR10_data
from minpy.nn.io import NDArrayIter

# Can also use MXNet IO here
# from mxnet.io import NDArrayIter

def main(args):
    data = get_CIFAR10_data(args.data_dir)
    # reshape all data to matrix
    data['X_train'] = data['X_train'].reshape(
        [data['X_train'].shape[0], 3 * 32 * 32])
    data['X_val'] = data['X_val'].reshape(
        [data['X_val'].shape[0], 3 * 32 * 32])
    data['X_test'] = data['X_test'].reshape(
        [data['X_test'].shape[0], 3 * 32 * 32])

    train_data = data['X_train']
    dataiter = NDArrayIter(
        data['X_train'], data['y_train'], batch_size=100, shuffle=True)

    count = 0
    for each_data in dataiter:
        print(each_data)
        count += 1
        if count == 10:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data iterator example")
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory that contains cifar10 data')
    main(parser.parse_args())
