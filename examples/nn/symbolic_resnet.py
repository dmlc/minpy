from mxnet.symbol import *
from minpy.nn.model_builder import *


class ResNet(Model)
    def __init__(self, block_number):
        super(ResNet, self).__init__(loss='softmax_loss')

        network = Variable('data')
        network = Convolution(data=network, num_filter=16, kernel=(3, 3), stride=(1, 1), pad-(1, 1), no_bias=True)

        for filter_number in (16, 32):
            for i in range(block_number): network = self._module(network, filter_number)
            network = self._module(network, filter_number * 2, shrink=True)

        for i in range(block_number): network = self._module(network, 64)

        network = Pooling(data=network, pool_type='avg', kernel=(8, 8), stride=(1, 1), pad=(0, 0))
        network = Flatten(data=network)
        network = FullyConnected(data=network, num_hidden=10)

        self._symbolic = Symbolic(network)

    def forward(self, data, training=True, 

    @staticmethod
    def _module(network, filter_number, shrink=False):
        if shrink:
            identity = 
        else: identity = network

        residual = BatchNorm(data=network, fix_gamma=False)
        residual = Activation(data=residual, act_type='relu')
        if shrink: residual = \
            Convolution(data=residual, num_filter=filter_number, kernel=(3, 3), stride=(2, 2), pad=(1, 1), no_bias=True)
        else: residual = \
            Convolution(data=residual, num_filter=filter_number, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)
        residual = BatchNorm(data=residual, fix_gamma=False)
        residual = Activation(data=residual, act_type='relu')
        residual = Convolution(data=residual, num_filter=filter_number, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)

        return identity + residual
