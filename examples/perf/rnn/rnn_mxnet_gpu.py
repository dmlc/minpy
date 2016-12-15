import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
from mxnet.io import NDArrayIter

from simple_rnn import rnn_unroll
import time

def data_gen(N, seq_len=30, high=1):
    """ A data generator for adding problem.

    The data definition strictly follows Quoc V. Le, Navdeep Jaitly, Geoffrey E.
    Hintan's paper, A Simple Way to Initialize Recurrent Networks of Rectified
    Linear Units.

    The single datum entry is a 2D vector with two rows with same length.
    The first row is a list of random data; the second row is a list of binary
    mask with all ones, except two positions sampled by uniform distribution.
    The corresponding label entry is the sum of the masked data. For
    example:
    
     input          label
     -----          -----
    1 4 5 3  ----->   9 (4 + 5)
    0 1 1 0

    :param N: the number of the entries.
    :param seq_len: the length of a single sequence.
    :param p: the probability of 1 in generated mask
    :param high: the random data is sampled from a [0, high] uniform distribution.
    :return: (X, Y), X the data, Y the label.
    """
    X_num = np.random.uniform(low=0, high=high, size=(N, seq_len, 1))
    X_mask = np.zeros((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in xrange(N):
        # Default uniform distribution on position sampling
        positions = np.random.choice(seq_len, size=2, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return X, Y

def L2(x, label):
    N = x.shape[0]
    C = x.shape[1]
    if len(label.shape) == 1:
        onehot_label = np.zeros([N, C])
        np.onehot_encode(label, onehot_label)
    else:
        onehot_label = label
    return np.sum((x - onehot_label) ** 2) / N


if __name__ == '__main__':
    #x_train, y_train = data_gen(10000)
    #x_test, y_test = data_gen(1000)

    x_train = np.random.rand(1000, 30, 256)
    y_train = np.random.rand(1000, 1)

    train_dataiter = NDArrayIter(x_train,
                                 y_train,
                                 batch_size=100,
                                 shuffle=True)
    '''
    test_dataiter = NDArrayIter(x_test,
                                y_test,
                                batch_size=100,
                                shuffle=False)
    '''
    
    for i in range(1, 11):
        num_rnn_layers = 1
        seq_len = 30
        input_size = 256
        num_hidden = 256 * i
        num_label = 1
        
        rnn_sym = rnn_unroll(num_rnn_layers, seq_len, input_size, num_hidden, num_label)

        context = mx.context.cpu()

        model = mx.model.FeedForward(ctx=context,
                                     symbol=rnn_sym,
                                     num_epoch=1,
                                     learning_rate=0.01,
                                     wd=0.0000,
                                     initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

        import logging
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=head)
        #model.fit(X=train_dataiter, eval_data=test_dataiter,eval_metric=mx.metric.np(L2))
        timers = [time.time()]
        def print_time(*args):
            print('Time: {}.'.format(time.time() - timers[0]))
            timers[0] = time.time()

        model.fit(X=train_dataiter, epoch_end_callback=print_time)

