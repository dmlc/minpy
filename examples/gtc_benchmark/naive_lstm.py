from time import time
import numpy as np
import mxnet as mx
import mxnet.contrib.autograd as autograd
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.nn.utils import cross_entropy


class NaiveLSTM(Model):
    def __init__(self, num_hidden):
        super(NaiveLSTM, self).__init__()

        self._num_hidden = num_hidden

        self._xi = FullyConnected(num_hidden=num_hidden)
        self._xf = FullyConnected(num_hidden=num_hidden)
        self._xo = FullyConnected(num_hidden=num_hidden)
        self._xg = FullyConnected(num_hidden=num_hidden)

        self._hi = FullyConnected(num_hidden=num_hidden)
        self._hf = FullyConnected(num_hidden=num_hidden)
        self._ho = FullyConnected(num_hidden=num_hidden)
        self._hg = FullyConnected(num_hidden=num_hidden)

        self._linear = FullyConnected(num_hidden=10)

    def _step(self, x, h, c):
        i = Sigmoid()(self._xi(x) + self._hi(h))
        f = Sigmoid()(self._xf(x) + self._hf(h))
        o = Sigmoid()(self._xo(x) + self._ho(h))
        g = Tanh()(self._xg(x) + self._hg(h))

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


def load_mnist(args):
    from joblib import load
    data = load(args.data_dir + 'mnist.dat')
    samples = 50000
    train_data, test_data = data['train_data'][:samples], data['test_data'][:samples]
    unpack_batch = lambda batch : (batch.data[0], batch.label[0])

    eps = 1e-5
    train_data = (train_data - train_data.mean(axis=0)) / (train_data.std(axis=0) + eps)
    test_data = (test_data - test_data.mean(axis=0)) / (test_data.std(axis=0) + eps)

    N, D = train_data.shape
    patch_size = 7
    sequence_length = D / patch_size
    train_data = train_data.reshape((N, sequence_length, patch_size))

    N, _ = test_data.shape
    test_data = test_data.reshape((N, sequence_length, patch_size))

    from mxnet.io import NDArrayIter
    batch_size = 64
    train_data_iter = NDArrayIter(train_data, data['train_label'][:samples], batch_size, shuffle=True)
    test_data_iter = NDArrayIter(test_data, data['test_label'][:samples], batch_size, shuffle=False)

    return train_data_iter, test_data_iter


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--rnn', type=str, default='RNN')
    args = parser.parse_args()

    from mxnet.context import Context
    context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
    Context.default_ctx = context

    unpack_batch = lambda batch : \
        (batch.data[0].as_in_context(context), batch.label[0].as_in_context(context))

    train_data_iter, test_data_iter = load_mnist(args)

    model = NaiveLSTM(128)
    updater = Updater(model, update_rule='sgd_momentum', lr=0.1, momentum=0.9)
    
    forward_time = 0
    backward_time = 0

    iteration_number = 0
    for epoch_number in range(50):
        for iteration, batch in enumerate(train_data_iter):
            iteration_number += 1

            data, labels = unpack_batch(batch)

            t0 = time()
            predictions = model.forward(data, is_train=True)
            forward_time += time() - t0

            loss = model.loss(predictions, labels, is_train=True)

            t0 = time()
            autograd.compute_gradient((loss,))
            backward_time += time() - t0

            updater(model.grad_dict)

            if iteration_number % 100 == 0:
                loss_value = cross_entropy(loss, labels)
                print 'iteration %d loss %f' % (iteration_number, loss_value)

                print forward_time, backward_time
                forward_time = 0
                backward_time = 0

        test_data_iter.reset()
        errors, samples = 0, 0
        for batch in test_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data)
            predictions = np.argmax(scores, axis=1)
            errors += np.count_nonzero(predictions - labels)
            samples += data.shape[0]

        print 'epoch %d validation error %f' % (epoch_number, errors / float(samples))
