import numpy as np
import mxnet as mx
import mxnet.contrib.autograd as autograd
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.nn.utils import cross_entropy
from examples.utils.data_utils import get_MNIST_data


class RNNModel(Model):
    def __init__(self, **kwargs):
        super(RNNModel, self).__init__()

        self._rnn = RNN(**kwargs)
        self._linear = FullyConnected(num_hidden=10)

    @Model.decorator
    def forward(self, data, mode='training'):
        hidden = self._rnn(data)
        return self._linear(hidden)

    @Model.decorator
    def loss(self, data, labels):
        return mx.nd.SoftmaxOutput(data, labels, normalization='batch')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--state_size', type=int, required=True)
    parser.add_argument('--patch', type=int, default=7)
    args = parser.parse_args()

    from mxnet.context import Context
    context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
    Context.default_ctx = context

    unpack_batch = lambda batch : \
        (batch.data[0].as_in_context(context), batch.label[0].as_in_context(context))

    train_data_iter, val_data_iter = get_MNIST_data(
        batch_size = args.batch_size,
        data_dir   = args.data_dir,
        normalize  = True,
        shape      = (784 // args.patch, args.patch),
    )

    kwargs = {attr : getattr(args, attr) for attr in ('state_size', 'num_layers', 'mode')}
    model = RNNModel(**kwargs)
    updater = Updater(model, update_rule='adam', lr=1e-3)
    
    iteration_number = 0
    for epoch_number in range(50):

        train_data_iter.reset()
        for iteration, batch in enumerate(train_data_iter):
            iteration_number += 1

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
