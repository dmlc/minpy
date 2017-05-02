'''
Deep Networks with Stochastic Depth
https://arxiv.org/abs/1603.09382
'''


from random import random
import numpy as np
import mxnet as mx
import mxnet.contrib.autograd as autograd
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.nn.utils import cross_entropy


class StochasticDepthNetwork(Model):
    def __init__(self, block_number, filter_numbers, p):
        super(StochasticDepthNetwork, self).__init__()

        self._block_number = block_number
        self._p = p

        f0, f1, f2 = filter_numbers

        # register blocks reducing size of feature maps
        self._shrinking_blocks = (
            Sequential(
                BatchNorm(fix_gamma=False),
                self._convolution(num_filter=f0),
            ),
            self._shrinking_block(f1),
            Sequential(
                self._shrinking_block(f2),
                BatchNorm(fix_gamma=False),
                ReLU(),
            )
        )

        # register blocks preserving size of feature maps
        self._residual_blocks = tuple(
            tuple(
                StochasticDepthNetwork._residual_block(f) \
                  for i in range(block_number)
            ) for f in filter_numbers
        )

        # compute class scores from feature maps
        self._to_scores = Sequential(
            Pooling(pool_type='avg', kernel=(8, 8), stride=(1, 1)),
            BatchFlatten(),
            FullyConnected(num_hidden=10),
        )
    
    @Model.decorator
    def forward(self, data):
        data = data

        for shrink, residual in zip(self._shrinking_blocks, self._residual_blocks):
            data = shrink(data)
            for index, r in enumerate(residual):
                p_block = 1 - (index + 1.0) / self._block_number * (1 - self._p)
                if random() < p_block:
                    data = data + r(data)

        return self._to_scores(data)

    @Model.decorator
    def loss(self, data, labels):
        return mx.nd.SoftmaxOutput(data, labels, normalization='batch')

    @staticmethod
    def _convolution(**kwargs):
        defaults = {'kernel' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True}
        defaults.update(kwargs)
        return Sequential(
            BatchNorm(fix_gamma=False),
            ReLU(),
            Convolution(**defaults),
        )

    @staticmethod
    def _residual_block(f):
        residual = Sequential(
            StochasticDepthNetwork._convolution(num_filter=f),
            StochasticDepthNetwork._convolution(num_filter=f),
        )
        return residual

    @staticmethod
    def _shrinking_block(f):
        return Sequential(
            Sequential(
                StochasticDepthNetwork._convolution(num_filter=f, stride=(2, 2)),
                StochasticDepthNetwork._convolution(num_filter=f),
            ) + \
            StochasticDepthNetwork._convolution(num_filter=f, kernel=(1, 1), stride=(2, 2), pad=(0, 0))
        )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    from load_cifar10_data_iter import *
    train_data_iter, val_data_iter = load_cifar10_data_iter(batch_size=128, path=args.data_dir)

    from mxnet.context import Context
    context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
    Context.default_ctx = context

    unpack_batch = lambda batch : \
        (batch.data[0].as_in_context(context), batch.label[0].as_in_context(context))

    model = StochasticDepthNetwork(16, (16, 32, 64), 0.1)
    updater = Updater(model, update_rule='adam', lr=1e-3)
    
    epoch_number = 0
    iteration_number = 0
    terminated = False

    while not terminated:
        epoch_number += 1

        train_data_iter.reset()
        for iteration, batch in enumerate(train_data_iter):
            iteration_number += 1
            if epoch_number > 500:
                terminated = True
                break
            if epoch_number in (250, 375):
                updater.lr = updater.lr * 0.1
               
            data, labels = unpack_batch(batch)
            predictions = model.forward(data, is_train=True)
            loss = model.loss(predictions, labels, is_train=True)
            autograd.compute_gradient((loss,))
            updater(model.grad_dict)

            if iteration_number % 100 == 0:
                loss_value = cross_entropy(loss, labels)
                print 'iteration %d loss %f' % (iteration_number, loss_value)
        
        val_data_iter.reset()
        errors, samples = 0, 0
        for batch in val_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data)
            predictions = mx.nd.argmax(scores, axis=1)
            errors += np.count_nonzero((predictions - labels).asnumpy())
            samples += data.shape[0]

        print 'epoch %d validation error %f' % (epoch_number, errors / float(samples))
