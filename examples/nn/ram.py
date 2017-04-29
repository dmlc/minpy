''' Recurrent Models of Visual Attention
    https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
'''

from scipy.misc import imresize as resize

from minpy.nn.model_builder import *
from minpy.nn.modules import *

class CoreNetwork(Model):
    def __init__(self):
        super(CoreNetwork, self).__init__()

        self._g_linear = FullyConnected(num_hidden=256)
        self._h_linear = FullyConnected(num_hidden=256)
        self._linear = FullyConnected(num_hidden=10)

    def forward(self, g, h, predict=False, **kwargs):
        if predict: return self._linear(h)
        elif h is None: return ReLU()(self._g_linear(g))
        else: return ReLU()(self._g_linear(g) + self._h_linear(h))


class GlimpseNetwork(Model):
    def __init__(self, length, n_patches):
        super(GlimpseNetwork, self).__init__()

        self._length = length
        self._n_patches = n_patches

        self._g_linear0 = FullyConnected(num_hidden=128)
        self._g_linear = FullyConnected(num_hidden=256)
        self._l_linear0 = FullyConnected(num_hidden=128)
        self._l_linear = FullyConnected(num_hidden=256)

    def forward(self, images, locations, mode='training'):
        if mode == 'training': self.training()
        elif mode == 'inference': self.inference()

        encoded = self._encode(images, locations, self._length, self._n_patches)
        h_g = self._g_linear0(encoded)
        h_g = ReLU()(h_g)
        h_g = self._g_linear(h_g)

        h_l = self._l_linear0(locations)
        h_l = ReLU(h_l)
        h_l = self._l_linear(h_l)

        return self._linear(h_g + h_l)

    @staticmethod
    def encode(images, locations, length, n_patches):
        N, H, V = images.shape
        locations[:, 0] = locations[:, 0] * H + H / 2
        locations[:, 1] = locations[:, 1] * V + V / 2

        d = length / 2
        images = np.pad(images, ((0, 0), (d, d), (d, d)), mode='edge')
        locations += d

        encoded = []

        for i in range(N):
            h_center, v_center = locations[i]
            h_from = h_center - d
            h_to = h_center + d
            v_from = v_center - d
            v_to = v_center + d
            
            image = images[i]
            l = length
            patches = []
            for p in range(n_patches):
                patch = image[h_from : h_to, v_from : v_to]
                resized = resize(patch, (length, length))
                reshaped = resized.reshape((1, length, length))
                patches.append(reshaped)
                l *= 2

            concatenated = np.concatenate(patches)
            reshaped = concatenated.reshape((1, n_patches, length, length))
            encoded.append(reshaped)
        
        return np.concatenate(encoded)

class LocationNetwork(Model):
    def __init__(self, variance):
        super(LocationNetwork, self).__init__()

        self._variance = variance
        self._linear = FullyConnected(num_hidden=2)

    def forward(self, h, **kwargs):
        locations = self._linear(h)
        return 

    def loss(self, locations, sampled, rewards):
        h_mu, v_mu = locations[:, 0], locations[:, 1]
        p = self.gaussian_pdf(sampled, h_mu, v_mu)
        return np.log(p) * rewards / len(sampled)

    @staticmethod
    def gaussian_pdf(X, h_mu, v_mu, sigma):
        rho = np.sum((X[:, 0] - h_mu)(X[:, 1] - v_mu)) / X.shape[0]
        z = ((X[:, 0] - h_mu) ** 2 - 2 * rho * (X[:, 0] - h_mu)(X[:, 1] - v_mu) + (X[:, 1] - v_mu) ** 2) / sigma ** 2
        return 1 / (2 * np.pi * sigma ** 2 * np.sqrt(1 - rho ** 2)) * np.exp(-z / (2 * (1 - rho **2)))
        
        
import numpy as np
images = np.ones((5, 32, 32))
h = np.random.choice(np.arange(32), (5, 1))
v = np.random.choice(np.arange(32), (5, 1))
locations = np.concatenate((h, v), axis=1)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--n_glimpses', type=int, required=True)
    args = parser.parse_args()

    core_network = CoreNetwork()
    glimpse_network = GlimpseNetwork()
    location_network = LocationNetwork()

    initial_locations = np.zeros((args.batch_size, 2)) # center

    unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())

    for iteration, batch in enumerate(train_data_iter):
        data, labels = unpack_batch(batch)
        g = glimpse_network(data, initial_locations)
        h = core_network(g, None)
        for i in range(args.n_glimpses - 1):
            l = location_network(h)
            g = glimpse_network(data, l)
            h = core_network(g)
