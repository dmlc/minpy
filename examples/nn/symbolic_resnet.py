from mxnet.symbol import *
import minpy.numpy as np
from minpy.nn.model_builder import Model, Updater
from minpy.nn.modules import Symbolic


class ResNet(Model):
    def __init__(self, block_number):
        super(ResNet, self).__init__(loss='softmax_loss')

        network = Variable('data')
        network = ResNet._convolution(data=network, num_filter=16, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

        for filter_number in (16, 32):
            for i in range(block_number): network = self._module(network, filter_number)
            network = self._module(network, filter_number * 2, shrink=True)

        for i in range(block_number): network = self._module(network, 64)

        network = BatchNorm(data=network, fix_gamma=False)

        network = Pooling(data=network, pool_type='avg', kernel=(8, 8), stride=(1, 1), pad=(0, 0))
        network = Flatten(data=network)
        network = FullyConnected(data=network, num_hidden=10)

        self._symbolic = Symbolic(network)

    def forward(self, data, training=True):
        if training: self.training()
        else: self.inference()

        return self._symbolic(data=data)

    @staticmethod
    def _convolution(**kwargs):
        return Convolution(no_bias=True, cudnn_tune='limited_workspace', **kwargs)

    @staticmethod
    def _module(network, filter_number, shrink=False):
        # TODO shrink
        if shrink: identity = \
            ResNet._convolution(data=network, num_filter=filter_number, kernel=(1, 1), stride=(2, 2), pad=(0, 0))
        else: identity = network

        residual = BatchNorm(data=network, fix_gamma=False)
        residual = Activation(data=residual, act_type='relu')
        stride = (2, 2) if shrink else (1, 1)
        residual = ResNet._convolution(data=residual, num_filter=filter_number, kernel=(3, 3), stride=stride, pad=(1, 1))
        residual = BatchNorm(data=residual, fix_gamma=False)
        residual = Activation(data=residual, act_type='relu')
        residual = ResNet._convolution(data=residual, num_filter=filter_number, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

        return identity + residual


if __name__ == '__main__':
    resnet = ResNet(3)

    updater = Updater(resnet, update_rule='sgd', learning_rate=0.1, momentem=0.9)

    from load_cifar10_data_iter import *
    train_data_iter, val_data_iter = load_cifar10_data_iter(batch_size=32)

    from minpy.context import set_context, gpu
    set_context(gpu(0))

    resnet.training()

    unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())

    for epoch in range(125):
        # anneal learning rate
        if epoch in (75, 100):
            updater.learning_rate = updater.learning_rate * 0.1
            print 'epoch %d learning rate annealed to %f' % (epoch, updater.learning_rate)

        # training
        train_data_iter.reset() # data iterator must be reset every epoch
        for iteration, batch in enumerate(train_data_iter):
            data, labels = unpack_batch(batch)
            resnet.forward(data) # TODO only forward once
            grad_dict, loss = resnet.grad_and_loss(data, labels)
            updater(grad_dict)
            if (iteration + 1) % 100 == 0:
                print 'epoch %d iteration %d loss %f' % (epoch, iteration + 1, loss[0])

        # validation
        val_data_iter.reset() # data iterator must be reset every epoch
        n_errors, n_samples = 0.0, 0.0
        for batch in val_data_iter:
            data, labels = unpack_batch(batch)
            probs = resnet.forward(data, True)
            preds = np.argmax(probs, axis=1)
            n_errors += np.count_nonzero(preds - labels)
            n_samples += len(data)

        print 'epoch %d validation error %f' % (epoch, n_errors / n_samples)
