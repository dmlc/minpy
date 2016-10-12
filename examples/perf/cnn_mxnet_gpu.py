import mxnet as mx
import argparse
import os
import sys
import time

batch_size = 256
num_epochs = 10


def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(
        data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(
        data=tanh2, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet


def main(args):
    img_fname = os.path.join(args.data_dir, 'train-images-idx3-ubyte')
    label_fname = os.path.join(args.data_dir, 'train-labels-idx1-ubyte')
    train = mx.io.MNISTIter(
        image=img_fname,
        label=label_fname,
        batch_size=batch_size,
        data_shape=(1, 28, 28))

    net = get_lenet()

    model = mx.model.FeedForward(
        symbol=net, num_epoch=num_epochs, learning_rate=1e-4, ctx=mx.gpu(0))
    timers = [time.time()]

    def print_time(*args):
        print('Time: {}.'.format(time.time() - timers[0]))
        timers[0] = time.time()

    model.fit(X=train, epoch_end_callback=print_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    main(parser.parse_args())
