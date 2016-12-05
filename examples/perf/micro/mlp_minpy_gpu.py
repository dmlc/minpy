"""Simple multi-layer perception neural network on MNIST."""
import argparse
import os.path
import struct
import time
import numpy as real_numpy

import minpy.numpy as np
from minpy.nn import io
from minpy.nn import layers
import minpy.nn.model
import minpy.nn.solver
from minpy.utils.minprof import minprof
# Please uncomment following if you have GPU-enabled MXNet installed.
from minpy.context import set_context, gpu
set_context(gpu(0)) # set the global context as gpu(0)

#import logging
#logging.getLogger('minpy.array').setLevel(logging.DEBUG)
#logging.getLogger('minpy.core').setLevel(logging.DEBUG)
#logging.getLogger('minpy.primitive').setLevel(logging.DEBUG)
#logging.getLogger('minpy.dispatch.policy').setLevel(logging.DEBUG)

num_cold = 5

class TwoLayerNet(minpy.nn.model.ModelBase):
    def __init__(self, args):
        super(TwoLayerNet, self).__init__()
        self.add_param(name='wi', shape=(784, args.hidden_size)) \
            .add_param(name='bi', shape=(args.hidden_size,))
        for i in range(args.num_hidden - 1):
            self.add_param(name='w%d' % i, shape=(args.hidden_size, args.hidden_size)) \
                .add_param(name='b%d' % i, shape=(args.hidden_size,))
        self.add_param(name='wo', shape=(args.hidden_size, 10)) \
            .add_param(name='bo', shape=(10,))
        self.batch_size = args.batch_size
        self.num_hidden = args.num_hidden

    def forward(self, X, mode):
        # Flatten the input data to matrix.
        X = np.reshape(X, (self.batch_size, 784))
        # First affine layer (fully-connected layer).
        y1 = layers.affine(X, self.params['wi'], self.params['bi'])
        # ReLU activation.
        y2 = layers.relu(y1)
        # Hidden layers.
        for i in range(self.num_hidden - 1):
            y2 = layers.affine(y2, self.params['w%d' % i], self.params['b%d' % i])
            y2 = layers.relu(y2)
        # Second affine layer.
        y3 = layers.affine(y2, self.params['wo'], self.params['bo'])
        return y3

    def loss(self, predict, y):
        # Compute softmax loss between the output and the label.
        return layers.softmax_loss(predict, y)


def main(args):
    # Create model.
    model = TwoLayerNet(args)
    for k, v in model.param_configs.items():
        model.params[k] = np.zeros(v['shape'])

    img = np.zeros((args.batch_size, 784))
    label = np.zeros((args.batch_size,))

    for l in range(args.num_loops):
        if l == num_cold:
            start = time.time()
        def loss_func(*params):
            f = model.forward(img, 'train')
            return model.loss(f, label)
        if args.only_forward:
            loss = loss_func()
            loss.asnumpy()
        else:
            param_arrays = list(model.params.values())
            param_keys = list(model.params.keys())
            grad_and_loss_func = minpy.core.grad_and_loss(
                loss_func, argnum=range(len(param_arrays)))
            grad_arrays, loss = grad_and_loss_func(*param_arrays)
            for g in grad_arrays:
                g.get_data(minpy.array_variants.ArrayType.MXNET).wait_to_read()
    dur = time.time() - start
    print('Per Loop Time: %.6f' % (dur / (args.num_loops - num_cold)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-forward', default=False, action='store_true')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--hidden-size', default=256, type=int)
    parser.add_argument('--num-hidden', default=1, type=int)
    parser.add_argument('--num-loops', default=20, type=int)
    #import profile
    #profile.run('main(parser.parse_args())')
    main(parser.parse_args())
