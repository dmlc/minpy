from time import time
import numpy as np
import mxnet as mx
import mxnet.contrib.autograd as autograd
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.nn.utils import cross_entropy
from examples.utils.data_utils import get_MNIST_data


class SophisticatedLSTM(Model):
    def __init__(self, num_hidden):
        super(SophisticatedLSTM, self).__init__()

        self._num_hidden = num_hidden

        self._x_linear = FullyConnected(num_hidden=num_hidden * 4)
        self._h_linear = FullyConnected(num_hidden=num_hidden * 4)

        self._linear = FullyConnected(num_hidden=10)

    def _step(self, x, h, c):
        stacked = self._x_linear(x) + self._h_linear(h)

        i = mx.nd.slice_axis(stacked, axis=1, begin=0, end=self._num_hidden)
        f = mx.nd.slice_axis(stacked, axis=1, begin=self._num_hidden, end=self._num_hidden * 2)
        o = mx.nd.slice_axis(stacked, axis=1, begin=self._num_hidden * 2, end=self._num_hidden * 3)
        g = mx.nd.slice_axis(stacked, axis=1, begin=self._num_hidden * 3, end=self._num_hidden * 4)

        i = Sigmoid()(i)
        f = Sigmoid()(f)
        o = Sigmoid()(o)
        g = Tanh()(g)

        c = f * c + i * g
        h = o * Tanh()(c)

        return h, c

    @Model.decorator
    def forward(self, data):
        N, L, D = data.shape

        h = mx.nd.zeros((N, self._num_hidden))
        c = mx.nd.zeros((N, self._num_hidden))

        for i in range(L):
            patch = mx.nd.slice_axis(data, axis=1, begin=i, end=(i + 1))
            h, c = self._step(patch, h, c)

        return self._linear(h)

    @Model.decorator
    def loss(self, data, labels):
        return mx.nd.SoftmaxOutput(data, labels, normalization='batch')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--num_hidden', type=int, required=True)
    parser.add_argument('--patch', type=int, default=7)
    args = parser.parse_args()

    from mxnet.context import Context
    context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
    Context.default_ctx = context

    unpack_batch = lambda batch : \
        (batch.data[0].as_in_context(context), batch.label[0].as_in_context(context))

    train_data_iter, test_data_iter = get_MNIST_data(
        batch_size = args.batch_size,
        data_dir   = args.data_dir,
        normalize  = True,
        shape      = (784 / args.patch, args.patch),
    )

    model = SophisticatedLSTM(args.num_hidden)
    updater = Updater(model, update_rule='sgd_momentum', lr=0.1, momentum=0.9)
    
    tft = 0 # training forward
    ift = 0 # inference forward
    bt = 0 # backward

    for i, batch in enumerate(train_data_iter):
        data, labels = unpack_batch(batch)

        t0 = time()
        predictions = model.forward(data, is_train=True)
        tft += time() - t0

        loss = model.loss(predictions, labels, is_train=True)

        t0 = time()
        autograd.compute_gradient((loss,))
        bt += time() - t0

        updater(model.grad_dict)

        if (i + 1) % 100 == 0:
            print tft, bt

    tft /= (i + 1)
    bt /= (i + 1)

    test_data_iter.reset()
    for i, batch in enumerate(test_data_iter):
        data, labels = unpack_batch(batch)
        
        t0 = time()
        scores = model.forward(data)
        ift += time() - t0

    print ift
    ift /= (i + 1)

    import cPickle as pickle
    pickle.dump((tft, ift, bt,), open('time/sophisticated-lstm-%d' % args.num_hidden, 'w'))
