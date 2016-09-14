Minpy IO
========

This tutorial introduces IO part of Minpy. We will describe the concepts of Dataset, DataIter and how to use MXNet IO module in Minpy code.

Dataset
-------

Dataset is a collection of samples. Each sample could have multiple entries, each represents a particular input variable for a learning task. For example, for image classification task, each sample should contain two entries, one is the input image, the other is the image label.

The source of a dataset could vary: a list of images for vision task, rows of text for NLP task, etc. The task of IO module is to turn the source dataset into the data structure that can be used by the learning system. In Minpy, the data structure for learning is Minpy.NDArray. Minpy could take numpy.ndarray, MXNet.NDArray and Minpy.NDArray as source dataset. These three kinds of source are easy to produce by our Minpy user in pure python code thus there’s no black box in preparing dataset. We can refer to `data_utils.py <https://github.com/dmlc/minpy/blob/master/examples/utils/data_utils.py>`_ to see how to prepare raw data for Minpy IO. 

If you want to utilize more complex IO schemas like prefetching, or handle raw data with decoding and augmentations, you can use the MXNet IO to achieve that. We will discuss it in later sessions.

DataIter
--------
Usually the optimization method for deep learning traverses data in a round-robin way. It perfectly match the pattern of python iterator. Minpy/MXNet both choose to implement IO logic into iterater. We can control the logic of each epoch by using:
::
    for each_batch in self.train_dataiter:

Generally to create a data iterator, you need to provide five kinds of parameters:

**Dataset Param** gives the basic information for the dataset, e.g. Minpy/Numpy/MXNet NDArray, file path, input shape, etc.

**Batch Param** gives the information to form a batch, e.g. batch size.

**Augmentation Param** tells which augmentation operations(e.g. crop, mirror) should be taken on an input image.

**Backend Param** controls the behavior of the backend threads to hide data loading cost.

**Auxiliary Param** provides options to help checking and debugging.

Usually, **Dataset Param** and **Batch Param** MUST be given, otherwise data batch can't be create. Other parameters can be given according to algorithm and performance need. Please check an example:
::
    train_dataiter = NDArrayIter(data=data['X_train'],
                                 label=data['y_train'],
                                 batch_size=batch_size,
                                 shuffle=True)

Using MXNet IO in Minpy
-----------------------

IO is a crucial part for deep learning. Raw data may need to go through a complex pipeline before feeding into solver and poor IO implementation could be the bottleneck for the whole system. MXNet has good practice on IO. Thus we would recommend referring to MXNet IO when you move on to complex task with performance requirement. 

To use MXNet IO, we just need to ``import mxnet.io`` then using the DataIters in there. For example:
::
    from mxnet.io import MNISTIter
    train           = MNISTIter(
        image       = data_dir + "train-images-idx3-ubyte",
        label       = data_dir + "train-labels-idx1-ubyte",
        input_shape = data_shape,
        batch_size  = args.batch_size,
        shuffle     = True,
        flat        = flat,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)


Minpy.NDArray can be constructed using MXNet.NDArray. So there’s nothing more we need to do on the rest of the code.

To get more information about MXNet IO, please visit `io.md <https://github.com/dmlc/mxnet/blob/master/docs/packages/python/io.md>`_ and `io.py <https://github.com/dmlc/mxnet/blob/master/python/mxnet/io.py>`_.





