"""Benchmark vanilla RNN using MinPy GPU."""
import argparse
import time

import minpy
import minpy.core as core
import minpy.numpy as np
from minpy.nn import io
from minpy.nn import layers
from minpy.nn.model import ModelBase
# Please uncomment following if you have GPU-enabled MXNet installed.
from minpy.context import set_context, gpu
set_context(gpu(0)) # set the global context as gpu(0)

import logging
logging.getLogger('minpy.array').setLevel(logging.DEBUG)
# logging.getLogger('minpy.core').setLevel(logging.DEBUG)
# logging.getLogger('minpy.primitive').setLevel(logging.DEBUG)

num_cold = 5

class RNNNet(ModelBase):
    def __init__(self, args):
        super(RNNNet, self).__init__()
        self.add_param(name='Wx', shape=(args.input_size, args.hidden_size)) \
            .add_param(name='Wh', shape=(args.hidden_size, args.hidden_size))\
            .add_param(name='b', shape=(args.hidden_size,))                  \
            .add_param(name='Wa', shape=(args.hidden_size, args.num_classes))\
            .add_param(name='ba', shape=(args.num_classes,))
        self.num_unroll_steps = args.num_unroll_steps
        self.hshape = (args.batch_size, args.hidden_size)

    def forward(self, X, mode):
        h = np.zeros(self.hshape)  # init hidden state
        for t in range(self.num_unroll_steps):
            h = layers.rnn_step(X, h, self.params['Wx'],
                                self.params['Wh'], self.params['b'])
        y = layers.affine(h, self.params['Wa'], self.params['ba'])
        return y

    def loss(self, predict, y):
        return layers.l2_loss(predict, y)


def main(args):
    # Create model.
    model = RNNNet(args)
    for k, v in model.param_configs.items():
        model.params[k] = np.zeros(v['shape'])

    data = np.zeros((args.batch_size, args.input_size)) # Data of only one time step.
    label = np.zeros((args.batch_size,))

    for l in range(args.num_loops):
        if l == num_cold:
            start = time.time()
        def loss_func(*params):
            f = model.forward(data, 'train')
            return model.loss(f, label)
        if args.only_forward:
            loss = loss_func()
            loss.wait_to_read()
        else:
            param_arrays = list(model.params.values())
            param_keys = list(model.params.keys())
            grad_and_loss_func = core.grad_and_loss(
                loss_func, argnum=range(len(param_arrays)))
            grad_arrays, loss = grad_and_loss_func(*param_arrays)
            for g in grad_arrays:
                g.wait_to_read()
        
    dur = time.time() - start
    print('Per Loop Time: %.6f' % (dur / (args.num_loops - num_cold)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-forward', default=False, action='store_true')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--hidden-size', default=512, type=int)
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--num-classes', default=10, type=int)
    parser.add_argument('--num-unroll-steps', default=30, type=int)
    parser.add_argument('--num-loops', default=20, type=int)
    #import profile
    #profile.run('main(parser.parse_args())')
    main(parser.parse_args())
