""" Simple multi-layer perception neural network using Minpy and MXNet symbols """
import argparse

import mxnet as mx
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy import core
from minpy.nn.io import NDArrayIter
# Can also use MXNet IO here
# from mxnet.io import NDArrayIter
from examples.utils.data_utils import get_CIFAR10_data

# Please uncomment following if you have GPU-enabled MXNet installed.
#from minpy.context import set_context, gpu
#set_context(gpu(0)) # set the global context as gpu(0)

batch_size=128
input_size=(3, 32, 32)
flattened_input_size=3 * 32 * 32
hidden_size=512
num_classes=10

class TwoLayerNet(ModelBase):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        # Use MXNet symbol to define the whole network.
        net = mx.sym.Variable(name='X')
        # Flatten the input data to matrix.
        net = mx.sym.Flatten(net)
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=hidden_size)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
        net = mx.sym.SoftmaxOutput(data=net, name='softmax', normalization='batch')
        # Wrap the final symbol into a function.
        # ATTENTION: when using mxnet symbols, input shape (including batch size) should be fixed
        input_shapes = {'X': (batch_size,) + input_size, 'softmax_label': (batch_size,)}
        self.fwd_fn = core.Function(net, input_shapes=input_shapes)
        # Add parameters.
        self.add_params(self.fwd_fn.get_params())

    def forward_batch(self, batch, mode):
        return self.fwd_fn(X=batch.data[0],
                           softmax_label=batch.label[0],
                           **self.params)

    def loss(self, predict, y):
        return layers.softmax_cross_entropy(predict, y)

def main(args):
    # Create model.
    model = TwoLayerNet()
    # Create data iterators for training and testing sets.
    data = get_CIFAR10_data(args.data_dir)
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
                        'learning_rate': 1e-4,
                        'momentum': 0.9
                    },
                    verbose=True,
                    print_every=20)
    # Initialize model parameters.
    solver.init()
    # Train!
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-layer perceptron example using mxnet "
                                                 "symbols")
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Directory that contains cifar10 data')
    main(parser.parse_args())
