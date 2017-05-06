import numpy as np
import mxnet as mx
import mxnet.contrib.autograd as autograd
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.nn.utils import cross_entropy


class VGG(Model):
    def __init__(self, layer_numbers, filter_numbers):
        super(VGG, self).__init__()
        self._blocks = tuple(VGG._block(l, f) for l, f in zip(layer_numbers, filter_numbers))

        fully_connected = lambda n : Sequential(
            FullyConnected(num_hidden=n),
            Dropout(p=0.5),
            ReLU()
        )

        self._to_scores = Sequential(
            VGG._pooling(),
            fully_connected(4096),
            fully_connected(4096),
            FullyConnected(num_hidden=1000)
        )
    
    @Model.decorator
    def forward(self, data):
        for block in self._blocks:
            data = block(data)

        return self._to_scores(data)

    @Model.decorator
    def loss(self, data, labels):
        return mx.nd.SoftmaxOutput(data, labels, normalization='batch')

    @staticmethod
    def _convolution(**kwargs):
        defaults = {'kernel' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True, 'cudnn_tune' : 'limited_workspace'}
        defaults.update(kwargs)
        return Sequential(Convolution(**defaults), ReLU())

    @staticmethod
    def _pooling(**kwargs):
        defaults = {'pool_type' : 'max', 'kernel' : (2, 2), 'stride' : (2, 2), 'pad' : (0, 0)}
        defaults.update(kwargs)
        return Pooling(**defaults)

    @staticmethod
    def _block(layer_number, filter_number):
        block = Sequential()
        for i in range(layer_number):
            block.append(VGG._convolution(num_filter=filter_number))
        block.append(VGG._pooling())

        return block


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir',type = str,required = True, help = 'Directory that contains cifar10 data')
    parser.add_argument('--gpu_index', type = int, default = 0)
    args = parser.parse_args()

    from mxnet.context import Context
    context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
    Context.default_ctx = context

    unpack_batch = lambda batch : \
        (batch.data[0].as_in_context(context), batch.label[0].as_in_context(context))

    from examples.utils.data_utils import get_imagenet_data
    train_data_iter, val_data_iter = get_imagenet_data(batch_size=16, path=args.data_dir)

    model = VGG((1, 1, 2, 2, 2),(16, 64, 128, 256, 256))
#   model = VGG((1, 1, 2, 2, 2),(64, 128, 256, 512, 512))
    updater = Updater(model, update_rule='sgd_momentum', lr=1e-3, momentum=0.9)

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
