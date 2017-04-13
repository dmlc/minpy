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
from minpy.utils.minprof import minprof

from minpy.context import set_context, gpu
set_context(gpu(1)) # set the global context as gpu(0)



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

class GenerativeNet(Model):
    def __init__(self, ngf, nc, no_bias):
        super(ResNet, self).__init__(loss='pass_top_grad_loss')
        self.layers = Sequential (
                _deconvolution_bn_relu(kernel=(4,4), num_filter=ngf*8, no_bias=no_bias),
                _deconvolution_bn_relu(kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*4, no_bias=no_bias),
                _deconvolution_bn_relu(kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2, no_bias=no_bias),
                _deconvolution_bn_relu(kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf, no_bias=no_bias),
                DeConvolution(kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias),
                Tanh(),
        )

    @staticmethod
    def _deconvolution_bn_relu(**kwargs):
        defaults = {'kernel' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True}
        defaults.update(kwargs)
        return Sequential(
            DeConvolution(**defaults),
            BatchNorm(fix_gamma=False),
            ReLU(),
        )

class DiscriminativeNet(Model):
    def __init__(self, ndf, no_bias):
        super(ResNet, self).__init__(loss='logistic_regression_loss')
        self.layers = Sequential(
                _convolution_leaky(kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf, no_bias=no_bias)
                _convolution_bn_leaky(kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2, no_bias=no_bias)
                _convolution_bn_leaky(kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4, no_bias=no_bias)
                _convolution_bn_leaky(kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8, no_bias=no_bias)
                Convolution(kernel=(4,4), num_filter=1, no_bias=no_bias)
                Flatten(),
        )

    @staticmethod
    def _convolution_bn_leaky(**kwargs):
        defaults = {'kernel' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True, act_type: 'leaky', 'slope' : 0.2}
        defaults.update(kwargs)
        return Sequential(
            Convolution(**defaults),
            BatchNorm(fix_gamma=False),
            LeakyReLU(**defaults),
        )

    @staticmethod
    def _convolution_leaky(**kwargs):
        defaults = {'kernel' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True, act_type: 'leaky', 'slope' : 0.2}
        defaults.update(kwargs)
        return Sequential(
            Convolution(**defaults),
            LeakyReLU(**defaults),
        )

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
    gnet_model = GenerativeNet(ngf, nc, no_bias)
    dnet_model = DiscriminativeNet(ndf, no_bias))
    
    # Prepare data
    X_train, X_test = get_mnist()
    real_iter = NDArrayIter(X_train, np.ones(X_train.shape[0]), batch_size=batch_size)
    rand_iter = RandIter(batch_size, Z)

    gnet_updater = Updater(gnet_model, update_rule='sgd', learning_rate=0.1, momentem=0.9)
    dnet_updater = Updater(dnet_model, update_rule='sgd', learning_rate=0.1, momentem=0.9)

    # Training    
    epoch_number = 0
    iteration_number = 0
    terminated = False

    while not terminated:
        # training
        epoch_number += 1
        train_data_iter.reset()
        
        for each_batch in self.real_dataiter:
            rand_batch = self.rand_dataiter.getdata()
            # train real
            dnet_real_grad_dict, dnet_real_loss = dnet_model.grad_and_loss(real_batch.data[0], real_batch.label[0])
            
            # train fake, by default, bp will not pass to the input of dnet
            generated_data = gnet.forward(rand_batch[0])
            fake_batch = DataBatch([generated_data], [np.zeros(generated_data.shape[0])])
            dnet_fake_grad_dict, dnet_fake_loss = dnet_model.grad_and_loss(fake_batch.data[0], fake_batch.label[0])
            
            # update dnet
            for each_key in dnet_real_grad_dict::
                dnet_real_grad_dict[each_key] += dnet_fake_grad_dict[each_key]
            dnet_updater(dnet_real_grad_dict)

            # ff dnet using fake data and real label
            fake_batch = DataBatch([generated_data], [np.ones(generated_data.shape[0])])
            dnet_model.add_to_grad(fake_batch.data[0])
            for each_param in dnet.params():
                dnet_model.remove_from_grad(each_param)
            dnet_grad_for_gnet, dnet_fake_as_real_loss = dnet_model.grad_and_loss(fake_batch.data[0], fake_batch.label[0])
            # backward take dnet_grad_for_gnet as label
            gnet_grad_dict, gnet_loss = gnet_model.backward(dnet_grad_for_gnet['input'])
            gnet_updater(gnet_grad_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Convolutional Generative Adversarial Net")
    main()
