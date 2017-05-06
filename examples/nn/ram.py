''' Recurrent Models of Visual Attention
    https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
'''


from math import pi
import numpy as np
from scipy.misc import imresize as resize
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.contrib.autograd as autograd
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.nn.utils import cross_entropy
from examples.utils.data_utils import get_MNIST_data


class CoreNetwork(Model):
    def __init__(self):
        super(CoreNetwork, self).__init__()

        self._g_linear = FullyConnected(num_hidden=256)
        self._h_linear = FullyConnected(num_hidden=256)
        self._linear = FullyConnected(num_hidden=10)

    @Model.decorator
    def forward(self, g=None, h=None):
        if g is None and h is None: return self._linear(h)
        elif h is None: return ReLU()(self._g_linear(g))
        else: return ReLU()(self._g_linear(g) + self._h_linear(h))


class GlimpseNetwork(Model):
    def __init__(self, size, n_patches):
        super(GlimpseNetwork, self).__init__()

        self._size = size
        self._n_patches = n_patches
        self._image_shape = (28, 28)

        self._g_linear0 = FullyConnected(num_hidden=128)
        self._g_linear = FullyConnected(num_hidden=256)
        self._l_linear0 = FullyConnected(num_hidden=128)
        self._l_linear = FullyConnected(num_hidden=256)

    @Model.decorator
    def forward(self, images, locations):
        images = images.reshape((images.shape[0],) + self._image_shape)
        encoded = self._encode(images, locations, self._size, self._n_patches)
        h_g = self._g_linear0(encoded)
        h_g = ReLU()(h_g)
        h_g = self._g_linear(h_g)

        h_l = self._l_linear0(locations)
        h_l = ReLU()(h_l)
        h_l = self._l_linear(h_l)

        return ReLU()(h_g + h_l)

    @staticmethod
    def _encode(images, locations, size, n_patches):
        images = images.asnumpy()
        locations = locations.asnumpy()

        N, H, V = images.shape
        locations[:, 0] = locations[:, 0] * H + H / 2
        locations[:, 1] = locations[:, 1] * V + V / 2

        d = size  / 2
        padding = (d * n_patches, d * n_patches)
        images = np.pad(images, ((0, 0), padding, padding), mode='edge')
        locations += d * n_patches

        encoded = []

        for i in range(N):
            h_center, v_center = locations[i]
            h_from = h_center - d
            h_to = h_center + d
            v_from = v_center - d
            v_to = v_center + d

            image = images[i]
            l = size
            patches = []
            for p in range(n_patches):
                print h_from, h_to, v_from, v_to
                patch = image[h_from : h_to, v_from : v_to]
                resized = resize(patch, (size, size))
                reshaped = resized.reshape((1, size, size))
                patches.append(reshaped)
                l *= 2

            concatenated = np.concatenate(patches)
            reshaped = concatenated.reshape((1, n_patches, size, size))
            encoded.append(reshaped)
        
        return nd.array(np.concatenate(encoded))

class LocationNetwork(Model):
    def __init__(self, sigma):
        super(LocationNetwork, self).__init__()

        self._sigma = sigma
        self._linear = FullyConnected(num_hidden=2)

    @Model.decorator
    def forward(self, h, **kwargs):
        return self._linear(h)

    @Model.decorator
    def loss(self, locations, sampled, rewards):
        p = self.gaussian_pdf(sampled, locations, self._sigma)
        return nd.mean(nd.log(p) * rewards)

    @staticmethod
    def gaussian_pdf(X, mu, sigma):
        h = nd.slice_axis(X, axis=1, begin=0, end=1)
        v = nd.slice_axis(X, axis=1, begin=1, end=2)
        h_mu = nd.slice_axis(mu, axis=1, begin=0, end=1)
        v_mu = nd.slice_axis(mu, axis=1, begin=1, end=2)

        z = ((h - h_mu) ** 2 - 2 * (h - h_mu) * (v - v_mu) + (v - v_mu) ** 2) / sigma ** 2
        return 1 / (2 * pi * sigma ** 2) * nd.exp(-z / 2)

    def sample(self, mu):
        # TODO context
        clip = lambda X : nd.minimum(1, nd.maximum(-1, X))

        sampled = nd.random_normal(shape=mu.shape).as_in_context(mu.context)
        return clip(sampled * self._sigma + mu)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--glimpse_size', type=int, default=4)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--n_glimpses', type=int, required=True)
    parser.add_argument('--n_patches', type=int, required=True)
    parser.add_argument('--sigma', type=float, default=0.001)
    args = parser.parse_args()

    core_network = CoreNetwork()
    glimpse_network = GlimpseNetwork(args.glimpse_size, args.n_patches)
    location_network = LocationNetwork(args.sigma)

    core_updater = Updater(core_network, update_rule='adam')
    glimpse_updater = Updater(glimpse_network, update_rule='adam')
    location_updater = Updater(location_network, update_rule='adam')

    from mxnet.context import Context
    context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
    Context.default_ctx = context

    unpack_batch = lambda batch : \
        (batch.data[0].as_in_context(context), batch.label[0].as_in_context(context))

    from examples.utils.data_utils import get_MNIST_data
    train_data_iter, test_data_iter = get_MNIST_data(batch_size=args.batch_size, data_dir=args.data_dir)

    initial_mu = nd.zeros((args.batch_size, 2)) # center

    for epoch in range(10):
        train_data_iter.reset()
        for iteration, batch in enumerate(train_data_iter):
            data, labels = unpack_batch(batch)

            history = []

            for i in range(args.n_glimpses):
                mu = initial_mu if i == 0 else location_network.forward(h, is_train=True)
                print mu.asnumpy()
                sampled = location_network.sample(mu)
                history.append((mu, sampled))

                g = glimpse_network.forward(data, sampled, is_train=True)
                h = core_network.forward(g, is_train=True)

            predictions = core_network.forward()

            cross_entropy_loss = cross_entropy(nd.SoftmaxOutput(predictions, labels, normalization='batch'))

            rewards = (nd.argmax(predictions, axis=1) == labels)
            reinforce_loss = sum(location_network.loss(mu, sampled, rewards) for mu, sampled in history)

            loss = cross_entropy_loss + reinforce_loss

            autograd.compute_gradient(loss)
            core_updater(core_network.grad_dict)
            glimpse_updater(glimpse_network.grad_dict)
            location_updater(location_network.grad_dict)

            if (iteration + 1) % 100 == 0:
                print 'iteration %d loss %f' % (iteration + 1, loss.asnumpy())
