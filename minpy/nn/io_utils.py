import mxnet as mx
import numpy as np

def getsubiter(dataiter, num_samples):
    """Create a sub dataiter which samples part of the data in the dataset"""
    idx = np.arange(dataiter.data[0][1].shape[0])
    np.random.shuffle(idx)
    mask = idx[0:num_samples]
    data = dataiter.data[0][1].asnumpy()[mask]
    label = dataiter.label[0][1].asnumpy()[mask]
    return mx.io.NDArrayIter(data, label, dataiter.batch_size, True)
