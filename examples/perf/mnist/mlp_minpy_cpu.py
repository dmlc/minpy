"""Simple multi-layer perception neural network on MNIST."""
import argparse
import os.path
import struct
import numpy as real_numpy

import minpy.numpy as np
from minpy.nn import io
from minpy.nn import layers
import minpy.nn.model
import minpy.nn.solver
import minpy.dispatch.policy
minpy.set_global_policy('only_numpy')

# import logging
# logging.getLogger('minpy.array').setLevel(logging.DEBUG)
# logging.getLogger('minpy.core').setLevel(logging.DEBUG)
# logging.getLogger('minpy.primitive').setLevel(logging.DEBUG)

batch_size = 256
flattened_input_size = 784
hidden_size = 256
num_classes = 10


class TwoLayerNet(minpy.nn.model.ModelBase):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.add_param(name='w1', shape=(flattened_input_size, hidden_size)) \
            .add_param(name='b1', shape=(hidden_size,)) \
            .add_param(name='w2', shape=(hidden_size, num_classes)) \
            .add_param(name='b2', shape=(num_classes,))

    def forward(self, X, mode):
        # Flatten the input data to matrix.
        X = np.reshape(X, (batch_size, flattened_input_size))
        # First affine layer (fully-connected layer).
        y1 = layers.affine(X, self.params['w1'], self.params['b1'])
        # ReLU activation.
        y2 = layers.relu(y1)
        # Second affine layer.
        y3 = layers.affine(y2, self.params['w2'], self.params['b2'])
        return y3

    def loss(self, predict, y):
        # Compute softmax loss between the output and the label.
        return layers.softmax_loss(predict, y)


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
        num_epochs=10,
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
