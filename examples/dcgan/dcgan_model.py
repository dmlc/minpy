from data_utils import get_mnist

import minpy
import minpy.numpy as np
import mxnet as mx
from minpy.nn.io import DataIter, NDArrayIter
import argparse

from dcgan_solver import DCGanSolver

from minpy.core import Function
from minpy.nn import layers
from minpy.nn.model import ModelBase
from examples.utils.data_utils import get_CIFAR10_data

# input shape
gnet_input_size = (100,1, 1)
nc = 3
ndf = 64
ngf = 64
dnet_input_size=(3, 64, 64)
batch_size = 64
Z = 100
lr = 0.0002
beta1 = 0.5
no_bias = True
fix_gamma = True
eps = 1e-5 + 1e-12

class GenerativeNet(ModelBase):
    def __init__(self):
        super(GenerativeNet, self).__init__()
        # Define symbols that using convolution and max pooling to extract better features
        BatchNorm = mx.sym.BatchNorm
        data = mx.sym.Variable('X')

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

        input_shapes = {'X': (batch_size,) + gnet_input_size}
        self.gnet = Function(gout, input_shapes=input_shapes, name='gnet')
        self.add_params(self.gnet.get_params())

    def forward_batch(self, batch_data, mode):
        out = self.gnet(X=batch_data, **self.params)
        return out
    
    # User get confused?
    def loss(self, dnet_bottom_gradient, predict):

        #print 'genet'
        #print predict

        return np.sum(dnet_bottom_gradient * predict)

class DiscriminativeNet(ModelBase):
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        BatchNorm = mx.sym.BatchNorm
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

        d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=2, no_bias=no_bias)
        dact_flat5 = mx.sym.Flatten(d5)
        #d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
        #dact5 = mx.sym.Activation(d5, name='dact5', act_type='sigmoid')
        #dact_flat5 = mx.sym.Flatten(dact5)
    
        input_shapes = {'data': (batch_size,) + dnet_input_size}
        self.dnet = Function(dact_flat5, input_shapes=input_shapes, name='dnet')
        self.add_params(self.dnet.get_params())

    def forward_batch(self, batch_data, mode):
        out = self.dnet(data=batch_data,
                **self.params)
        return out

    def loss(self, predict, y):
        #return layers.logistic_cross_entropy(predict, y)
        return layers.softmax_loss(predict, y)

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = [np.zeros(batch_size)]

    def iter_next(self):
        return True

    def getdata(self):
        return [np.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

def main():
    # Create model.
    gnet_model = GenerativeNet()
    dnet_model = DiscriminativeNet()
    
    # Prepare data
    X_train, X_test = get_mnist()
    train_iter = NDArrayIter(X_train, np.ones(X_train.shape[0]), batch_size=batch_size)
    rand_iter = RandIter(batch_size, Z)

    # Create solver.
    solver = DCGanSolver(gnet_model,
                    dnet_model,
                    train_iter,
                    rand_iter,
                    num_epochs=100,
                    init_rule='gaussian',
                    init_config={
                        'stdvar': 0.02
                    },
                    update_rule='adam',
                    optim_config={
                        'learning_rate': lr,
                        'wd': 0.,
                        'beta1': beta1,
                    },
                    verbose=True,
                    print_every=20)
    # Initialize model parameters.
    solver.init()
    # Train!
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Convolutional Generative Adversarial Net")
    main()

