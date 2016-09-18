MinPy IO
========

This tutorial introduces IO part of MinPy. We will describe the concepts of Dataset, DataIter and how to use MXNet IO module in MinPy code.

Dataset
-------

Dataset is a collection of samples. Each sample may have multiple entries, each representing a particular input variable for a learning task. For example, for image classification task, each sample may contain one entry for the image, and another for the label.

The source of a dataset can vary: a list of images for vision task, rows of text for NLP task, etc. The task of the IO module is to turn the source dataset into the data structure that can be used by the learning system. In MinPy, the data structure for learning is ``minpy.NDArray``, whereas ``numpy.ndarray``, ``mxnet.NDArray`` and ``minpy.NDArray`` can all serve as source dataset. These three kinds of source are easy to produce in pure Python code, thus there’s no black box in preparing the dataset. Please refer to `data_utils.py <https://github.com/dmlc/minpy/blob/master/examples/utils/data_utils.py>`_ to see how to prepare raw data for MinPy IO.

If you want to utilize more complex IO schemas like prefetching, or handle raw data with decoding and augmentations, you can use the MXNet IO to achieve that, as we will discuss it later.

DataIter
--------
Usually the optimization method for deep learning traverses data in a round-robin fashion. This perfectly matches the pattern of Python iterator. Therefore, MinPy/MXNet both choose to implement IO logic with iterater.

Generally, to create a data iterator, you need to provide five kinds of parameters:

* **Dataset Param** gives the basic information for the dataset, e.g. MinPy/NumPy/MXNet NDArray, file path, input shape, etc.
* **Batch Param** gives the information to form a batch, e.g. batch size.
* **Augmentation Param** tells which augmentation operations(e.g. crop, mirror) should be taken on an input image.
* **Backend Param** controls the behavior of the backend threads in order to hide data loading cost.
* **Auxiliary Param** provides options to help checking and debugging.

Usually, **Dataset Param** and **Batch Param** MUST be given, otherwise data batch cannot be created. Other parameters can be given according to algorithm and performance need. Suppose the raw source dataset has been prepared into a dictionary``data``, with two entries, containing training input and label, then the following code materialize them into an ``NDArray`` iterator:
::
    train_dataiter = NDArrayIter(data=data['X_train'],
                                 label=data['y_train'],
                                 batch_size=batch_size,
                                 shuffle=True)

We can now control the logic of each epoch by using:
::
    for each_batch in self.train_dataiter:
    
Refer to the ``_step`` function in `solver.py <https://github.com/dmlc/minpy/blob/master/minpy/nn/solver.py>`_ to see how data are accessed.
    
Using MXNet IO in MinPy
-----------------------

IO is a crucial part for deep learning. Raw data may need to go through a complex pipeline before feeding into solver. Obviously, poor IO implementation can become performance bottleneck. MXNet has a high performing and mature IO subsystem, we recommend MXNet IO when you move on to complex task and/or need better performance. 

Minpy has automatically importd all the available MXNet DataIter into minpy.io namespace. To use MXNet IO, we just need to ``import minpy.io`` then using the DataIters. For example:
::
    from minpy.io import MNISTIter
    train           = MNISTIter(
        image       = data_dir + "train-images-idx3-ubyte",
        label       = data_dir + "train-labels-idx1-ubyte",
        input_shape = data_shape,
        batch_size  = args.batch_size,
        shuffle     = True,
        flat        = flat,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

..

Current available MXNet DataIters include: ``MNISTIter, ImageRecordIter, CSVIter, PrefetchingIter, ResizeIter``. To get more information about MXNet IO, please visit `io.md <https://github.com/dmlc/mxnet/blob/master/docs/packages/python/io.md>`_ and `io.py <https://github.com/dmlc/mxnet/blob/master/python/mxnet/io.py>`_.

Note
----
The above is in the context of supervised learning. Of course, this is not the only game in town. For example, `Policy gradient reinforcement learning <http://minpy.readthedocs.io/en/latest/rl_policy_gradient/rl_policy_gradient.html>`_ is an interesting departure. 




