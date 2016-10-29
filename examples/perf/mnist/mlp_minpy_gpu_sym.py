"""Simple multi-layer perception neural network on MNIST."""
import argparse
import os.path
import struct
import numpy as real_numpy

import mxnet as mx
import minpy
import minpy.numpy as np
from minpy.nn import io
from minpy.nn import layers
import minpy.nn.model
import minpy.nn.solver
from minpy import context
context.set_context(context.gpu(0))

# import logging
# logging.getLogger('minpy.array').setLevel(logging.DEBUG)
# logging.getLogger('minpy.core').setLevel(logging.DEBUG)
# logging.getLogger('minpy.primitive').setLevel(logging.DEBUG)

batch_size = 128
flattened_input_size = 784
hidden_size = 256
num_classes = 10


class TwoLayerNet(minpy.nn.model.ModelBase):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        # Use MXNet symbol to define the whole network.
        net = mx.sym.Variable(name='X')
        # Flatten the input data to matrix.
        net = mx.sym.Flatten(net)
        net = mx.sym.FullyConnected(
            data=net, name='fc1', num_hidden=hidden_size)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.FullyConnected(
            data=net, name='fc2', num_hidden=num_classes)
        net = mx.sym.SoftmaxOutput(net, name='softmax', normalization='batch')
        # Wrap the final symbol into a function.
        input_shapes={'X': (batch_size, flattened_input_size), 'softmax_label': (batch_size,)}
        self.fwd_fn = minpy.core.Function(net, input_shapes=input_shapes)
        # Add parameters.
        self.add_params(self.fwd_fn.get_params())

    def forward_batch(self, batch, mode):
        return self.fwd_fn(X=batch.data[0],
                           softmax_label=batch.label[0],
                           **self.params)

    def loss(self, predict, y):
        # Compute softmax loss between the output and the label.
        return layers.softmax_cross_entropy(predict, y)


def main(args):
    # Create model.
    model = TwoLayerNet()
    # Create data iterators for training and testing sets.
    img_fname = os.path.join(args.data_dir, 'train-images-idx3-ubyte')
    label_fname = os.path.join(args.data_dir, 'train-labels-idx1-ubyte')
    with open(label_fname, 'rb') as f:
        magic_nr, size = struct.unpack('>II', f.read(8))
        assert magic_nr == 2049
        assert size == 60000
        label = real_numpy.fromfile(f, dtype=real_numpy.int8)
    with open(img_fname, 'rb') as f:
        magic_nr, size, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic_nr == 2051
        assert size == 60000
        assert rows == cols == 28
        img = real_numpy.fromfile(
            f, dtype=real_numpy.uint8).reshape(size, rows * cols)

    train_dataiter = io.NDArrayIter(
        data=img, label=label, batch_size=batch_size, shuffle=True)

    # Create solver.
    solver = minpy.nn.solver.Solver(
        model,
        train_dataiter,
        train_dataiter,
        num_epochs=20,
        init_rule='gaussian',
        init_config={'stdvar': 0.001},
        update_rule='sgd_momentum',
        optim_config={'learning_rate': 1e-4,
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
