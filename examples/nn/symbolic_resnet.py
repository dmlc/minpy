from mxnet.symbol import *
import minpy.numpy as np
from minpy.nn.model_builder import Model, Updater
from minpy.nn.modules import Symbolic


class ResNet(Model):
    def __init__(self, block_number):
        super(ResNet, self).__init__(loss='softmax_loss')

        network = Variable('data')
        network = ResNet._convolution(data=network, num_filter=16)

        for filter_number in (16, 32):
            for i in range(block_number): network = self._module(network, filter_number)
            network = self._module(network, filter_number * 2, shrink=True)

        for i in range(block_number): network = self._module(network, 64)

        network = Activation(data=network, act_type='relu')
        network = BatchNorm(data=network, fix_gamma=False)
        network = Pooling(data=network, pool_type='avg', kernel=(8, 8), stride=(1, 1), pad=(0, 0))
        network = Flatten(data=network)
        network = FullyConnected(data=network, num_hidden=10)

        self._symbolic = Symbolic(network)

    def forward(self, data, mode='training'):
        if mode == 'training': self.training()
        elif mode == 'inference': self.inference()

        return self._symbolic(data=data)

    @staticmethod
    def _convolution(**kwargs):
        defaults = {'kernel' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True}
        defaults.update(kwargs)
        return Convolution(**defaults)

    @staticmethod
    def _module(network, f, shrink=False):
        # TODO shrink
        if shrink: identity = \
            ResNet._convolution(data=network, num_filter=f, kernel=(1, 1), stride=(2, 2), pad=(0, 0))
        else: identity = network

        residual = BatchNorm(data=network, fix_gamma=False)
        residual = Activation(data=residual, act_type='relu')
        stride = (2, 2) if shrink else (1, 1)
        residual = ResNet._convolution(data=residual, num_filter=f, stride=stride)
        residual = BatchNorm(data=residual, fix_gamma=False)
        residual = Activation(data=residual, act_type='relu')
        residual = ResNet._convolution(data=residual, num_filter=f)

        return identity + residual


unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    from load_cifar10_data_iter import *
    train_data_iter, val_data_iter = load_cifar10_data_iter(batch_size=128, path=args.data_dir)

    '''
    from examples.utils.data_utils import get_CIFAR10_data
    data = get_CIFAR10_data(args.data_dir)
    '''

    '''
    from minpy.nn.io import NDArrayIter
    batch_size = 128
    train_data_iter = NDArrayIter(data=data['X_train'], label=data['y_train'], batch_size=batch_size, shuffle=True)
    val_data_iter = NDArrayIter(data=data['X_test'], label=data['y_test'], batch_size=batch_size, shuffle=False)
    '''

    from minpy.context import set_context, gpu
    set_context(gpu(args.gpu_index))

    model = ResNet(3)
    updater = Updater(model, update_rule='sgd', learning_rate=0.1, momentem=0.9)
    
    epoch_number = 0
    iteration_number = 0
    terminated = False

    while not terminated:
        # training
        epoch_number += 1
        train_data_iter.reset()

        for iteration, batch in enumerate(train_data_iter):
            iteration_number += 1
            if iteration_number > 64000:
                terminated = True
                break
            if iteration_number in (32000, 48000):
                updater.learning_rate = updater.learning_rate * 0.1
               
            data, labels = unpack_batch(batch)
            grad_dict, loss = model.grad_and_loss(data, labels)
            updater(grad_dict)

            if iteration_number % 100 == 0:
                print 'iteration %d loss %f' % (iteration_number, loss)

        # validation
        print model.aux_params

        val_data_iter.reset()
        errors, samples = 0, 0
        for batch in val_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data, 'inference') # TODO training=False
            predictions = np.argmax(scores, axis=1)
            errors += np.count_nonzero(predictions - labels)
            samples += len(data)

        print 'epoch %d validation error %f' % (epoch_number, errors / float(samples))
