""" Simple multi-layer perception neural network using Caffe Op through minpy and MXNet symbols """
import sys
import mxnet as mx
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from examples.utils.data_utils import get_CIFAR10_data
from minpy import core
from minpy.nn.io import NDArrayIter
# Can also use MXNet IO here
# from mxnet.io import NDArrayIter

class TwoLayerCaffeNet(ModelBase):
    def __init__(self,
                 input_size=3 * 32 * 32,
                 hidden_size=512,
                 num_classes=10):
        super(TwoLayerCaffeNet, self).__init__()
        # ATTENTION: mxnet's weight dimension arrangement is different; it is [out_size, in_size]
        self.param_configs['w1'] = { 'shape': [hidden_size, input_size] }
        self.param_configs['b1'] = { 'shape': [hidden_size,] }
        self.param_configs['w2'] = { 'shape': [num_classes, hidden_size] }
        self.param_configs['b2'] = { 'shape': [num_classes,] }
        # define the symbols
        data = mx.sym.Variable(name='X')
        fc1 = mx.sym.CaffeOp(name='fc1',
                             data_0=data,
                             num_weight=2,
                             prototxt="layer {type:\"InnerProduct\" inner_product_param{num_output: %d} }"%hidden_size)
        act = mx.sym.CaffeOp(data_0=fc1,
                             prototxt="layer {type:\"ReLU\"}")
        fc2 = mx.sym.CaffeOp(name='fc2',
                             data_0=act,
                             num_weight=2,
                             prototxt="layer {type:\"InnerProduct\" inner_product_param{num_output: %d} }"%num_classes)
        # ATTENTION: when using mxnet symbols, input shape (including batch size) should be fixed
        self.fwd_fn = core.Function(fc2, {'X': (100, input_size)})

    def forward(self, X, mode):
        return self.fwd_fn(X=X,
                           fc1_0_weight=self.params['w1'],
                           fc1_1_bias=self.params['b1'],
                           fc2_0_weight=self.params['w2'],
                           fc2_1_bias=self.params['b2'])

    def loss(self, predict, y):
        return layers.softmax_loss(predict, y)

def main(_):
    model = TwoLayerCaffeNet()
    data = get_CIFAR10_data()
    # reshape all data to matrix
    data['X_train'] = data['X_train'].reshape([data['X_train'].shape[0], 3 * 32 * 32])
    data['X_val'] = data['X_val'].reshape([data['X_val'].shape[0], 3 * 32 * 32])
    data['X_test'] = data['X_test'].reshape([data['X_test'].shape[0], 3 * 32 * 32])
    # ATTENTION: the batch size should be the same as the input shape declared above.
    train_dataiter = NDArrayIter(data['X_train'],
                         data['y_train'],
                         100,
                         True)

    test_dataiter = NDArrayIter(data['X_test'],
                         data['y_test'],
                         100,
                         True)
    solver = Solver(model,
                    train_dataiter,
                    test_dataiter,
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
