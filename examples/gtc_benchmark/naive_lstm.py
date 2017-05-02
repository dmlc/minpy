from time import time
import numpy as np
import mxnet as mx
import mxnet.contrib.autograd as autograd
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.nn.utils import cross_entropy
from examples.utils.data_utils import get_MNIST_data

def sigmoid(x):
    return .5 * (mx.nd.tanh(.5 * x) + 1)

class NaiveLSTM(Model):
    def __init__(self, num_hidden):
        super(NaiveLSTM, self).__init__()

        self._num_hidden = num_hidden

        self._xi = FullyConnectedND(num_hidden=num_hidden)
        self._xf = FullyConnectedND(num_hidden=num_hidden)
        self._xo = FullyConnectedND(num_hidden=num_hidden)
        self._xg = FullyConnectedND(num_hidden=num_hidden)

        self._hi = FullyConnectedND(num_hidden=num_hidden)
        self._hf = FullyConnectedND(num_hidden=num_hidden)
        self._ho = FullyConnectedND(num_hidden=num_hidden)
        self._hg = FullyConnectedND(num_hidden=num_hidden)

        self._linear = FullyConnectedND(num_hidden=10)

    def _step(self, x, h, c):
        i = sigmoid(self._xi(x) + self._hi(h))
        f = sigmoid(self._xf(x) + self._hf(h))
        o = sigmoid(self._xo(x) + self._ho(h))
        g = mx.nd.tanh(self._xg(x) + self._hg(h))

        c = f * c + i * g
        h = o * mx.nd.tanh(c)

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

    #@Model.decorator
    #def loss(self, data, labels):
        #return mx.nd.SoftmaxOutput(data, labels, normalization='batch')
    @Model.decorator
    def loss(self, data, labels):
        N, _ = data.shape
        return mx.nd.sum((data - labels) ** 2) / N

def print_readable_loss(logits, labels):
    N, C = logits.shape
    logits_np = logits.asnumpy()
    labels_np = labels.asnumpy().astype(np.int)
    loss = (-np.sum(np.log(logits_np[np.arange(N), labels_np])) / N)
    print('Loss %.6f' % loss)

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
        shape      = (784 // args.patch, args.patch),
    )

    model = NaiveLSTM(args.num_hidden)
    updater = Updater(model, update_rule='sgd_momentum', lr=0.1, momentum=0.9)
    
    tft = 0 # training forward
    ift = 0 # inference forward
    bt = 0 # backward

    for i, batch in enumerate(train_data_iter):
        data, labels = unpack_batch(batch)

        t0 = time()
        predictions = model.forward(data, is_train=True)
        tft += time() - t0

        #loss = model.loss(predictions, labels, is_train=True)
        #print_readable_loss(loss, labels)

        onehot_labels = mx.nd.one_hot(labels, 10)
        loss = model.loss(predictions, onehot_labels, is_train=True)
        print('Loss %.6f' % loss.asnumpy())

        t0 = time()
        autograd.compute_gradient((loss,))
        for k, v in model.grad_dict.items():
            v.wait_to_read()
        bt += time() - t0

        updater(model.grad_dict)

        if (i + 1) % 20 == 0:
            print('Per batch forward time %.3f. Per batch backward time %.3f' % (tft / 20, bt / 20))
            tft = 0
            bt = 0

    tft /= (i + 1)
    bt /= (i + 1)

    test_data_iter.reset()
    for i, batch in enumerate(test_data_iter):
        data, labels = unpack_batch(batch)

        t0 = time()
        scores = model.forward(data)
        ift += time() - t0

    ift /= (i + 1)

    import cPickle as pickle
    pickle.dump((tft, ift, bt,), open('time/naive-lstm-%d' % args.num_hidden, 'w'))
