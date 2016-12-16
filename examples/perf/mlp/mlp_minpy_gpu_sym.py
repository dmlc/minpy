import argparse
import os.path
import struct
import time

import mxnet as mx

import minpy.core as core
import minpy.numpy as np
from minpy.nn import io
from minpy.nn import layers
import minpy.nn.model
import minpy.nn.solver
# Please uncomment following if you have GPU-enabled MXNet installed.
from minpy.context import set_context, gpu
set_context(gpu(0)) # set the global context as gpu(0)

import logging
logging.getLogger('minpy.array').setLevel(logging.DEBUG)
#logging.getLogger('minpy.core').setLevel(logging.DEBUG)
#logging.getLogger('minpy.primitive').setLevel(logging.DEBUG)

num_cold = 5

class TwoLayerNet(minpy.nn.model.ModelBase):
    def __init__(self, args):
        super(TwoLayerNet, self).__init__()
        self.batch_size = args.batch_size
        # Use MXNet symbol to define the whole network.
        net = mx.sym.Variable(name='X')
        # Flatten the input data to matrix.
        net = mx.sym.Flatten(net)
        net = mx.sym.FullyConnected(data=net, num_hidden=args.hidden_size)
        net = mx.sym.Activation(data=net, act_type='relu')
        for i in range(args.num_hidden - 1):
            net = mx.symbol.FullyConnected(data=net, num_hidden=args.hidden_size)
            net = mx.symbol.Activation(data=net, act_type="relu")
        net = mx.sym.FullyConnected(data=net, num_hidden=10)
        net = mx.sym.SoftmaxOutput(net, name='softmax', normalization='batch')
        # Wrap the final symbol into a function.
        input_shapes={'X': (args.batch_size, 784),
                      'softmax_label': (args.batch_size,)}
        self.fwd_fn = minpy.core.Function(net, input_shapes=input_shapes)
        # Add parameters.
        self.add_params(self.fwd_fn.get_params())

    def forward_batch(self, batch, mode):
        return self.fwd_fn(X=batch.data[0],
                           softmax_label=batch.label[0],
                           **self.params)

    def loss(self, predict, y):
        # Compute softmax loss between the output and the label.
        return layers.softmax_cross_entropy(predict, y)


def main(args):
    # Create model.
    model = TwoLayerNet(args)
    for k, v in model.param_configs.items():
        model.params[k] = np.zeros(v['shape'])
    param_keys = list(model.params.keys())

    img = np.zeros((args.batch_size, 784))
    label = np.zeros((args.batch_size,))
    batch = minpy.nn.io.DataBatch([img], [label])

    for l in range(args.num_loops):
        if l == num_cold:
            start = time.time()
        def loss_func(*params):
            f = model.forward_batch(batch, 'train')
            return model.loss(f, label)
        if args.only_forward:
            loss = loss_func()
            loss.get_data(minpy.array_variants.ArrayType.MXNET).wait_to_read()
        else:
            param_arrays = list(model.params.values())
            grad_and_loss_func = core.grad_and_loss(
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
