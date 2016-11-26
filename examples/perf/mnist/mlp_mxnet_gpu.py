"""Simple multi-layer perception neural network on MNIST."""
import argparse
import os.path
import time

import mxnet as mx

batch_size = 256
flattened_input_size = 784
hidden_size = 256
num_classes = 10
num_epochs = 10


def main(args):
    # Create data iterators for training and testing sets.
    img_fname = os.path.join(args.data_dir, 'train-images-idx3-ubyte')
    label_fname = os.path.join(args.data_dir, 'train-labels-idx1-ubyte')
    train = mx.io.MNISTIter(
        image=img_fname,
        label=label_fname,
        batch_size=batch_size,
        data_shape=(784, ))
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data=data, num_hidden=hidden_size)
    act1 = mx.symbol.Activation(data=fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=act1, num_hidden=10)
    mlp = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    model = mx.model.FeedForward(
        symbol=mlp, num_epoch=num_epochs, learning_rate=1e-4, ctx=mx.gpu(0))
    timers = [time.time()]

    def print_time(*args):
        print('Time: {}.'.format(time.time() - timers[0]))
        timers[0] = time.time()

    model.fit(X=train, epoch_end_callback=print_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    main(parser.parse_args())
