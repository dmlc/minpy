import argparse
import os.path
import sys
import time
import mxnet as mx

data_shape = (3, 28, 28)
batch_size = 128
num_epochs = 10
hidden_size = 512
num_classes = 10


def get_net():
    input_data = mx.symbol.Variable(name='data')
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(7, 7), num_filter=32)
    relu1 = mx.symbol.Activation(data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type='max', kernel=(2, 2), stride=(2, 2))

    flatten = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=hidden_size)
    relu6 = mx.symbol.Activation(data=fc1, act_type='relu')
    fc2 = mx.symbol.FullyConnected(data=relu6, num_hidden=num_classes)

    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return softmax


def main(args):
    train = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.data_dir, 'train.rec'),
        mean_img=os.path.join(args.data_dir, 'mean.bin'),
        data_shape=data_shape,
        batch_size=batch_size,
        rand_crop=True)
    net = get_net()

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
