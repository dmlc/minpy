""" Simple multi-layer perception neural network using Minpy and MXNet symbols """
import sys
import minpy
import numpy as np
import numpy.random as npr
import mxnet as mx
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy.utils.data_utils import get_CIFAR10_data
from minpy import core

class TwoLayerNet(ModelBase):
    def __init__(self,
                 input_size=3 * 32 * 32,
                 hidden_size=512,
                 num_classes=10):
        super(TwoLayerNet, self).__init__()
        # ATTENTION: mxnet's weight dimension arrangement is different; it is [out_size, in_size]
        self.param_configs['w1'] = { 'shape': [hidden_size, input_size] }
        self.param_configs['b1'] = { 'shape': [hidden_size,] }
        self.param_configs['w2'] = { 'shape': [num_classes, hidden_size] }
        self.param_configs['b2'] = { 'shape': [num_classes,] }
        # define the symbols
        data = mx.sym.Variable(name='X')
        fc1 = mx.sym.FullyConnected(name='fc1',
                                    data=data,
                                    num_hidden=hidden_size)
        act = mx.sym.Activation(data=fc1, act_type='relu')
        fc2 = mx.sym.FullyConnected(name='fc2',
                                    data=act,
                                    num_hidden=num_classes)
        # ATTENTION: when using mxnet symbols, input shape (including batch size) should be fixed
        self.fwd_fn = core.function(fc2, {'X': (128, input_size)})

    def forward(self, X):
        return self.fwd_fn(X=X,
                           fc1_weight=self.params['w1'],
                           fc1_bias=self.params['b1'],
                           fc2_weight=self.params['w2'],
                           fc2_bias=self.params['b2'])

    def loss(self, predict, y):
        return layers.softmax_loss(predict, y)

def main(_):
    model = TwoLayerNet()
    data = get_CIFAR10_data()
    # reshape all data to matrix
    data['X_train'] = data['X_train'].reshape([data['X_train'].shape[0], 3 * 32 * 32])
    data['X_val'] = data['X_val'].reshape([data['X_val'].shape[0], 3 * 32 * 32])
    data['X_test'] = data['X_test'].reshape([data['X_test'].shape[0], 3 * 32 * 32])
    # ATTENTION: the batch size should be the same as the input shape declared above.
    solver = Solver(model,
                    data,
                    num_epochs=10,
                    batch_size=128,
                    init_rule='xavier',
                    update_rule='sgd_momentum',
                    optim_config={
                        'learning_rate': 1e-4,
                        'momentum': 0.9
                    },
                    verbose=True,
                    print_every=20)
    solver.init()
    solver.train()

if __name__ == '__main__':
    main(sys.argv)
