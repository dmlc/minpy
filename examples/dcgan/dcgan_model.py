import minpy
import minpy.numpy as np
import mxnet as mx

from minpy.core import Function
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from examples.utils.data_utils import get_CIFAR10_data

class GenerativeNet(ModelBase):
    def __init__(self):
        super(GenerativeNet, self).__init__()
        # Define symbols that using convolution and max pooling to extract better features
        BatchNorm = mx.sym.BatchNorm
        data = mx.sym.Variable('data')

        g1 = mx.sym.Deconvolution(data, name='g1', kernel=(4,4), num_filter=ngf*8, no_bias=no_bias)
        gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
        gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

        g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*4, no_bias=no_bias)
        gbn2 = BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
        gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

        g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2, no_bias=no_bias)
        gbn3 = BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
        gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

        g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf, no_bias=no_bias)
        gbn4 = BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
        gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

        g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias)
        gout = mx.sym.Activation(g5, name='gact5', act_type='tanh')

        input_shapes = {'X': (batch_size,) + input_size}
        self.gnet = Function(net, input_shapes=input_shapes, name='gnet')
        self.add_params(self.gnet.get_params())

    def forward_batch(self, batch, mode):
        out = self.gnet(X=batch.data[0], **self.params)
        return out
    
    # User get confused?
    def loss(self, dnet_bottom_gradient, predict):
        return np.sum(dnet_bottom_gradient * predict)

class DiscrimiinativeNet(ModelBase):
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')

        d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf, no_bias=no_bias)
        dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

        d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2, no_bias=no_bias)
        dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
        dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

        d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4, no_bias=no_bias)
        dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
        dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

        d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8, no_bias=no_bias)
        dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
        dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

        d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
        d5 = mx.sym.Flatten(d5)
    
        input_shapes = {'data': (batch_size,) + input_size}
        self.dnet = Function(d5, input_shapes=input_shapes, name='dnet')
        self.add_params(self.dnet.get_params())

    def forward_batch(self, batch, mode):
        out = self.dnet(data=batch.data[0],
                **self.params)
        return out

    def loss(self, predict, y):
        return layers.softmax_cross_entropy(predict, y)

def main(args):
    # Create model.
    gnet_model = GenerativeNet()
    dnet_model = DiscrimiinativeNet()
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
    solver = GanSolver(gnet_model,
                    dnet_model,
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

