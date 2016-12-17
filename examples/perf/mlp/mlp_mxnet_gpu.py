"""Simple multi-layer perception neural network on MNIST."""
import argparse
import os.path
import time

import mxnet as mx

num_cold = 5

def main(args):
    # Create data iterators for training and testing sets.
    net = mx.symbol.Variable('data')
    net = mx.symbol.FullyConnected(data=net, num_hidden=args.hidden_size)
    net = mx.symbol.Activation(data=net, act_type="relu")
    for i in range(args.num_hidden - 1):
        net = mx.symbol.FullyConnected(data=net, num_hidden=args.hidden_size)
        net = mx.symbol.Activation(data=net, act_type="relu")
    net = mx.symbol.FullyConnected(data=net, num_hidden=10)
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')

    arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=(args.batch_size, 784))
    arg_types, out_types, aux_types = net.infer_type(data=mx.base.mx_real_t)

    arg_arrays = [mx.nd.zeros(shape, mx.gpu(0), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    grad_dict = {name : mx.nd.zeros(shape, mx.gpu(0), dtype=dtype)
                 for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                 if name != 'data'}

    executor = net.bind(ctx=mx.gpu(0),
                        args=arg_arrays,
                        args_grad=grad_dict,
                        grad_req='write')

    for i in range(args.num_loops):
        if i == num_cold:
            start = time.time()
        outputs = executor.forward()
        if args.only_forward:
            for o in outputs:
                o.wait_to_read()
            continue
        executor.backward([outputs[0]])
        for name, grad in grad_dict.items():
            grad.wait_to_read()
    dur = time.time() - start
    print('Per Loop Time: %.6f' % (dur / (args.num_loops - num_cold)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-forward', default=False, action='store_true')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--hidden-size', default=256, type=int)
    parser.add_argument('--num-hidden', default=1, type=int)
    parser.add_argument('--num-loops', default=20, type=int)
    main(parser.parse_args())
