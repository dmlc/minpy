import mxnet as mx
import numpy as np
import six.moves.cPickle as pickle


def getsubiter(dataiter, num_samples):
    """Create a sub dataiter which samples part of the data in the dataset"""
    idx = np.arange(dataiter.data[0][1].shape[0])
    np.random.shuffle(idx)
    mask = idx[0:num_samples]
    data = dataiter.data[0][1].asnumpy()[mask]
    label = dataiter.label[0][1].asnumpy()[mask]
    return mx.io.NDArrayIter(data, label, dataiter.batch_size, True)


def save_data_labels(X, Y, file_name):
    """ Handy utility to save data
    :param X: data vector
    :param Y: corresponding label vector
    :param file_name: file saved to
    """
    with open(file_name, 'wb') as f:
        data = {}
        data['data'] = X
        data['labels'] = Y
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_labels(file_name):
    """ Handy utility to unpack data
    :param file_name: file to unpack
    :return: (X, Y), (data vector, label vector)
    """
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        X = data['data']
        Y = data['labels']
        return X, Y
