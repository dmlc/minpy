import numpy as ori_np
import minpy.numpy as np
import minpy.nn.io as io
import os, gzip
import pickle as pickle
import time
import sys

def test_NDArrayIter():
    # check minpy.ndarray as input
    datas = np.ones([1000, 2, 2])
    labels = np.ones([1000, 1])
    for i in range(1000):
        datas[i] = i / 100
        labels[i] = i / 100
    dataiter = io.NDArrayIter(datas, labels, batch_size = 100, shuffle = True, last_batch_handle='pad')
    batchidx = 0
    for batch in dataiter:
        batchidx += 1
    assert(batchidx == 10)
    dataiter = io.NDArrayIter(datas, labels, batch_size = 100, shuffle = False, last_batch_handle='pad')
    batchidx = 0
    labelcount = [0 for i in range(10)]
    for batch in dataiter:
        label = batch.label[0].asnumpy().flatten()
        assert((batch.data[0].asnumpy()[:,0,0] == label).all())
        for i in range(label.shape[0]):
            labelcount[int(label[i])] += 1
    for i in range(10):
        assert(labelcount[i] == 100)
    
    # check numpy.ndarray as input
    datas = ori_np.ones([1000, 2, 2])
    labels = ori_np.ones([1000, 1])
    for i in range(1000):
        datas[i] = i / 100
        labels[i] = i / 100
    dataiter = io.NDArrayIter(datas, labels, batch_size = 100, shuffle = True, last_batch_handle='pad')
    batchidx = 0
    for batch in dataiter:
        batchidx += 1
    assert(batchidx == 10)
    dataiter = io.NDArrayIter(datas, labels, batch_size = 100, shuffle = False, last_batch_handle='pad')
    batchidx = 0
    labelcount = [0 for i in range(10)]
    for batch in dataiter:
        label = batch.label[0].asnumpy().flatten()
        assert((batch.data[0][:,0,0].asnumpy() == label).all())
        for i in range(label.shape[0]):
            labelcount[int(label[i])] += 1
    for i in range(10):
        assert(labelcount[i] == 100)

    # check padding
    datas = np.ones([1000, 2, 2])
    labels = np.ones([1000, 1])
    for i in range(1000):
        datas[i] = i / 100
        labels[i] = i / 100
    dataiter = io.NDArrayIter(datas, labels, batch_size = 128, shuffle = True, last_batch_handle='pad')
    batchidx = 0
    for batch in dataiter:
        batchidx += 1
    assert(batchidx == 8)
    dataiter = io.NDArrayIter(datas, labels, batch_size = 128, shuffle = False, last_batch_handle='pad')
    batchidx = 0
    labelcount = [0 for i in range(10)]
    for batch in dataiter:  
        label = batch.label[0].asnumpy().flatten()
        assert((batch.data[0].asnumpy()[:,0,0] == label).all())
        for i in range(label.shape[0]):
            labelcount[int(label[i])] += 1

    for i in range(10):
        if i == 0:
            assert(labelcount[i] == 124)
        else:
            assert(labelcount[i] == 100)

    # check padding
    datas = ori_np.ones([1000, 2, 2])
    labels = ori_np.ones([1000, 1])
    for i in range(1000):
        datas[i] = i / 100
        labels[i] = i / 100
    dataiter = io.NDArrayIter(datas, labels, batch_size = 128, shuffle = True, last_batch_handle='pad')
    batchidx = 0
    for batch in dataiter:
        batchidx += 1
    assert(batchidx == 8)
    dataiter = io.NDArrayIter(datas, labels, batch_size = 128, shuffle = False, last_batch_handle='pad')
    batchidx = 0
    labelcount = [0 for i in range(10)]
    for batch in dataiter:  
        label = batch.label[0].asnumpy().flatten()
        assert((batch.data[0].asnumpy()[:,0,0] == label).all())
        for i in range(label.shape[0]):
            labelcount[int(label[i])] += 1

    for i in range(10):
        if i == 0:
            assert(labelcount[i] == 124)
        else:
            assert(labelcount[i] == 100)

if __name__ == "__main__":
    test_NDArrayIter()
