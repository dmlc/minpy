""" Simple multi-layer perception neural network using Minpy """
import argparse

import minpy.numpy as np
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
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
        self.add_param(name='w1', shape=(flattened_input_size, hidden_size)) \
            .add_param(name='b1', shape=(hidden_size,)) \
            .add_param(name='w2', shape=(hidden_size, num_classes)) \
            .add_param(name='b2', shape=(num_classes,)) \
            .add_param(name='gamma', shape=(hidden_size,), init_rule='constant', init_config={'value': 1.0}) \
            .add_param(name='beta', shape=(hidden_size,), init_rule='constant') \
            .add_aux_param(name='running_mean', value=None) \
            .add_aux_param(name='running_var', value=None)

    def forward(self, X, mode):
        # Flatten the input data to matrix.
        X = np.reshape(X, (batch_size, 3 * 32 * 32))
        # First affine layer (fully-connected layer).
        y1 = layers.affine(X, self.params['w1'], self.params['b1'])
        # ReLU activation.
        y2 = layers.relu(y1)
        # Batch normalization
        y3, self.aux_params['running_mean'], self.aux_params['running_var'] = layers.batchnorm(
            y2, self.params['gamma'], self.params['beta'], running_mean=self.aux_params['running_mean'], \
            running_var=self.aux_params['running_var'])
        # Second affine layer.
        y4 = layers.affine(y3, self.params['w2'], self.params['b2'])
        # Dropout
        y5 = layers.dropout(y4, 0.5, mode=mode)
        return y5

    def loss(self, predict, y):
        # Compute softmax loss between the output and the label.
        return layers.softmax_loss(predict, y)


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
    parser = argparse.ArgumentParser(description="Multi-layer perceptron example using minpy operators")
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Directory that contains cifar10 data')
    main(parser.parse_args())
