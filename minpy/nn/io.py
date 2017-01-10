""" DataIter for convenient IO. """
from __future__ import division
from __future__ import absolute_import
from collections import OrderedDict

import sys
import inspect
import mxnet.io
import six.moves.cPickle as pickle # pylint: disable=import-error, no-name-in-module
import numpy as np

from .. import array

class DataBatch(object): # pylint: disable=too-few-public-methods
    """Default object for holding a mini-batch of data and related information."""

    def __init__(self, data, label, pad=None, index=None):
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index


class DataIter(object):
    """DataIter object in mxnet. """

    def __init__(self):
        self.batch_size = 0

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator. """
        pass

    def next(self):
        """Get next data batch from iterator. Equivalent to
        self.iter_next()
        DataBatch(self.getdata(), self.getlabel(), self.getpad(), None)

        Returns
        -------
        data : DataBatch
            The data of next batch.
        """
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Iterate to next batch.

        Returns
        -------
        has_next : boolean
            Whether the move is successful.
        """
        pass

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """
        pass

    def getlabel(self):
        """Get label of current batch.

        Returns
        -------
        label : NDArray
            The label of current batch.
        """
        pass

    def getindex(self):
        """Get index of the current batch.

        Returns
        -------
        index : numpy.array
            The index of current batch
        """
        pass

    def getpad(self):
        """Get the number of padding examples in current batch.

        Returns
        -------
        pad : int
            Number of padding examples in current batch
        """
        pass


def _init_data(data, allow_empty, default_name):
    # pylint: disable=invalid-name, redefined-variable-type
    """Convert data into canonical form."""
    assert (data is not None) or allow_empty
    if data is None:
        data = []

    if isinstance(data, (np.ndarray, array.Array)):
        data = [data]
    if isinstance(data, list):
        if not allow_empty:
            assert len(data) > 0
        if len(data) == 1:
            data = OrderedDict([(default_name, data[0])])
        else:
            data = OrderedDict([('_%d_%s' % (i, default_name), d)
                                for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError(
            "Input must be NDArray, numpy.ndarray, MinPy Array, or "
            "a list of them or dict with them as values.")
    for k, v in data.items():
        if not isinstance(v, (np.ndarray, array.Array)):
            raise TypeError(("Invalid type '%s' for %s, " % (type(
                v), k)) + "should be NDArray, numpy.ndarray, or MinPy Array.")

    return list(data.items())


class NDArrayIter(DataIter):
    # pylint: disable=too-many-instance-attributes, no-member
    """NDArrayIter object in minpy. Taking numpy array to get dataiter.
    Parameters
    ----------
    data: numpy.ndarray, a list of them, or a dict of string to them.
        NDArrayIter supports single or multiple data and label.
    label: numpy.ndarray, a list of them, or a dict of them.
        Same as data, but is not fed to the model during testing.
    batch_size: int
        Batch Size
    shuffle: bool
        Whether to shuffle the data
    last_batch_handle: 'pad', 'discard' or 'roll_over'
        How to handle the last batch
    Note
    ----
    This iterator will pad, discard or roll over the last batch if
    the size of data does not match batch_size. Roll over is intended
    for training and can cause problems if used for prediction.
    """

    def __init__(self,
                 data,
                 label=None,
                 batch_size=1,
                 shuffle=False,
                 last_batch_handle='pad'):
        # pylint: disable=W0201, too-many-arguments

        super(NDArrayIter, self).__init__()

        self.data = _init_data(data, allow_empty=False, default_name='data')
        self.label = _init_data(
            label, allow_empty=True, default_name='softmax_label')

        # shuffle data
        if shuffle:
            idx = np.arange(self.data[0][1].shape[0])
            np.random.shuffle(idx)
            self.data = [(k, v[idx]) for k, v in self.data]
            self.label = [(k, v[idx]) for k, v in self.label]

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)
        self.num_iterations_per_epoch = self.data[0][1].shape[0] / batch_size

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data_list[0].shape[0] - self.data_list[0].shape[
                0] % batch_size
            data_dict = OrderedDict(self.data)
            label_dict = OrderedDict(self.label)
            for k, _ in self.data:
                data_dict[k] = data_dict[k][:new_n]
            for k, _ in self.label:
                label_dict[k] = label_dict[k][:new_n]
            self.data = list(data_dict.items())
            self.label = list(label_dict.items())
        self.num_data = self.data_list[0].shape[0]
        assert self.num_data >= batch_size, \
            "batch_size need to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:])))
                for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:])))
                for k, v in self.label]

    def hard_reset(self):
        """Igore roll over data and set to start"""
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor % self.num_data) % self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        if self.iter_next():
            return DataBatch(
                data=self.getdata(),
                label=self.getlabel(),
                pad=self.getpad(),
                index=None)
        else:
            raise StopIteration

    def _getdata(self, data_source):
        # pylint: disable=line-too-long
        """Load data from underlying arrays, internal use only"""
        assert (self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            if isinstance(data_source[0][1], array.Array):
                return [x[1][self.cursor:self.cursor + self.batch_size] for x in data_source]
            elif isinstance(data_source[0][1], np.ndarray):
                return [array.wrap(x[1][self.cursor:self.cursor + self.batch_size]) for x in data_source]
            else:
                raise TypeError("Invalid data type, only numpy.ndarray and minpy.array.Array are allowed.")
        else:
            pad = self.batch_size - self.num_data + self.cursor
            if isinstance(data_source[0][1], array.Array):
                return [array.wrap(np.concatenate((x[1][self.cursor:].asnumpy(), x[1][:pad].asnumpy()), axis=0))
                        for x in data_source]
            elif isinstance(data_source[0][1], np.ndarray):
                return [array.wrap(np.concatenate((x[1][self.cursor:], x[1][:pad]), axis=0))
                        for x in data_source]
            else:
                raise TypeError("Invalid data type, only numpy.ndarray and minpy.array.Array are allowed.")


    def getdata(self):
        return self._getdata(self.data)

    def getlabel(self):
        return self._getdata(self.label)

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0

    def getindex(self):
        return self.cursor / self.batch_size

    def getsubiter(self, num_samples):
        """Create a sub dataiter which samples part of the data in the dataset"""
        idx = np.arange(self.data[0][1].shape[0])
        np.random.shuffle(idx)
        mask = idx[0:num_samples]
        data = [v[mask] for _, v in self.data]
        label = [v[mask] for _, v in self.label]
        return NDArrayIter(data, label, self.batch_size, True)

    def getnumiterations(self):
        """Get how many iterations per epoch"""
        return self.num_iterations_per_epoch


def save_data_labels(data_vec, label_vec, file_name):
    """ Handy utility to save data

    :param data_vec: data vector
    :param label_vec: corresponding label vector
    :param file_name: file saved to
    """
    with open(file_name, 'wb') as save_file:
        data = {}
        data['data'] = data_vec
        data['labels'] = label_vec
        pickle.dump(data, save_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_labels(file_name):
    """ Handy utility to unpack data

    :param file_name: file to unpack
    :return: (data_vec, label_vec), (data vector, label vector)
    """
    with open(file_name, 'rb') as load_file:
        data = pickle.load(load_file)
        data_vec = data['data']
        label_vec = data['labels']
        return data_vec, label_vec


def _import_mxnetio():
    module_obj = sys.modules[__name__]
    member_dict = dict((cls[0], cls[1])
                       for cls in inspect.getmembers(module_obj))
    # mxnet python io is class members
    clsmembers = inspect.getmembers(mxnet.io, inspect.isclass)
    for cls in clsmembers:
        if cls[0].endswith("Iter") and (not cls[0] in member_dict):
            setattr(module_obj, cls[0], cls[1])
    # mxnet c++ io is function members
    funmembers = inspect.getmembers(mxnet.io, inspect.isfunction)
    for fun in funmembers:
        if fun[0].endswith("Iter") and (not fun[0] in member_dict):
            setattr(module_obj, fun[0], fun[1])

# Import mxnet python io into minpy namespace
_import_mxnetio()
