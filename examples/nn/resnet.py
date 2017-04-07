import minpy.numpy as np

from minpy.nn.model_builder import *
from minpy.nn.modules import *


class ResNet(Model):
    # TODO prettify convolution interface (current using mxnet interface)
    def __init__(self, block_number, filter_numbers):
        super(ResNet, self).__init__(loss='softmax_loss')

        # blocks preserving input dimensionality
        preserving_blocks = []
        for filter_number in filter_numbers:
            sequential = Sequential()
            for i in range(block_number):
                sequential.append(ResNet._preserving_block(filter_number))
            preserving_blocks.append(sequential)

        self._preserving_blocks = preserving_blocks # register

        # blocks that reduce size of feature maps but increase number of filters
        shrinking_blocks = [ResNet._convolution(num_filter=filter_numbers[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1))]

        for filter_number in filter_numbers[1:]:
            shrinking_blocks.append(ResNet._shrinking_block(filter_number))

        self._shrinking_blocks = shrinking_blocks # register

        # compute class scores from feature maps
        self._to_scores = Sequential(
            Pooling(pool_type='avg', kernel=(8, 8), stride=(1, 1), pad=(0, 0)),
            Flatten(),
            FullyConnected(num_hidden=10),
        )
    
    def forward(self, data, training=True):
        if training: self.training()
        else: self.inference()

        out = data

        for shrinking_block, preserving_block in \
            zip(self._shrinking_blocks, self._preserving_blocks):
            out = shrinking_block(out)
            out = preserving_block(out)

        out = self._to_scores(out)

        return out

    @staticmethod
    def _convolution(**kwargs):
        return Sequential(
            Convolution(no_bias=True, **kwargs),
            ReLU(),
            BatchNorm(),
        )

    @staticmethod
    def _preserving_block(filter_number):
        # a block preserving input dimensionality
        identity = Identity()
        residual = Sequential(
            ResNet._convolution(num_filter=filter_number, kernel=(3, 3), stride=(1, 1), pad=(1, 1)),
            ResNet._convolution(num_filter=filter_number, kernel=(3, 3), stride=(1, 1), pad=(1, 1)),
        )
        return identity + residual

    @staticmethod
    def _shrinking_block(filter_number):
        return Sequential(
            ResNet._convolution(num_filter=filter_number, kernel=(3, 3), stride=(2, 2), pad=(1, 1)),
            Convolution(num_filter=filter_number, kernel=(1, 1), stride=(1, 1), pad=(0, 0)) \
                + \
            ResNet._convolution(num_filter=filter_number, kernel=(3, 3), stride=(1, 1), pad=(1, 1)),
        )


if __name__ == '__main__':
    import sys
    sys.settrace

    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--gpu_index', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    '''

    resnet = ResNet(3, (16, 32, 64))

    updater = Updater(resnet, update_rule='sgd', learning_rate=0.1, momentem=0.9)

    from load_cifar10_data_iter import *
    train_data_iter, val_data_iter = load_cifar10_data_iter(batch_size=64)

    '''
    from mxnet.io import NDArrayIter
    import numpy as np
    train_data_iter = NDArrayIter(np.random.normal(0, 1, (50000, 3, 32, 32)), np.random.choice(np.arange(10), (50000,)))
    val_data_iter = NDArrayIter(np.random.normal(0, 1, (10000, 3, 32, 32)), np.random.choice(np.arange(10), (10000,)))
    import minpy.numpy as np
    '''

    from minpy.context import set_context, gpu
    set_context(gpu(0))

    resnet.training()

    unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())

    for epoch in range(125):
        # TODO errors
        # *** Error in `/usr/bin/python': double free or corruption (fasttop): 0x00007f96f8004830 ***
        # *** Error in `/usr/bin/python': free(): invalid pointer: 0x00007fbecda65760 ***
        # Segmentation fault (core dumped)

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
            probs = resnet.forward(data)
            preds = np.argmax(probs, axis=1)
            n_errors += np.count_nonzero(preds - labels)
            n_samples += len(data)

        print 'epoch %d validation error %f' % (epoch, n_errors / n_samples)
