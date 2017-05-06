import numpy as np
import mxnet as mx
import mxnet.contrib.autograd as autograd
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.nn.utils import cross_entropy


class ResNet(Model):
    def __init__(self, block_number, filter_numbers):
        super(ResNet, self).__init__()

        f0, f1, f2 = filter_numbers

        # register blocks reducing size of feature maps
        self._shrinking_blocks = (
            Convolution(
                num_filter = f0,
                kernel     = (3, 3),
                stride     = (1, 1),
                pad        = (1, 1),
                no_bias    = True,
            ),
            self._shrinking_block(f1),
            self._shrinking_block(f2),
        )

        # register blocks preserving size of feature maps
        self._residual_blocks = tuple(
            tuple(
                ResNet._residual_block(f) \
                  for i in range(block_number)
            ) for f in filter_numbers
        )

        # compute class scores from feature maps
        self._to_scores = Sequential(
            BatchNorm(fix_gamma=False),
            ReLU(),
            Pooling(pool_type='avg', kernel=(8, 8), stride=(1, 1)),
            BatchFlatten(),
            FullyConnected(num_hidden=10),
        )
    
    @Model.decorator
    def forward(self, data):
        data = data

        for shrink, residual in zip(self._shrinking_blocks, self._residual_blocks):
            data = shrink(data)
            for r in residual:
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
            ResNet._convolution(num_filter=f),
            ResNet._convolution(num_filter=f),
        )
        return residual

    @staticmethod
    def _shrinking_block(f):
        return Sequential(
            ResNet._convolution(num_filter=f, stride=(2, 2)),
            ResNet._convolution(num_filter=f),
        ) + ResNet._convolution(num_filter=f, kernel=(1, 1), stride=(2, 2), pad=(0, 0))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    from mxnet.context import Context
    context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
    Context.default_ctx = context

    unpack_batch = lambda batch : \
        (batch.data[0].as_in_context(context), batch.label[0].as_in_context(context))

    # TODO data iterator ending batch issue
    from load_cifar10_data_iter import *
    train_data_iter, val_data_iter = load_cifar10_data_iter(batch_size=128, path=args.data_dir)
    
    model = ResNet(3, (16, 32, 64))
    updater = Updater(model, update_rule='sgd_momentum', lr=0.1, momentum=0.9)
    
    epoch_number = 0
    iteration_number = 0
    terminated = False

    while not terminated:
        epoch_number += 1

        train_data_iter.reset()
        for iteration, batch in enumerate(train_data_iter):
            iteration_number += 1
            if iteration_number > 64000:
                terminated = True
                break
            if iteration_number in (32000, 48000):
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
