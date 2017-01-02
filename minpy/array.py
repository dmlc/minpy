#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=unused-argument, protected-access, logging-format-interpolation, abstract-method
# pylint: disable=pointless-string-statement
"""Base type for arrays."""
from __future__ import absolute_import
from __future__ import print_function

import itertools
import collections

import mxnet
import numpy

from .array_variants import ArrayType
from .array_variants import array_types
from .array_variants import number_types
from .context import current_context
from .utils import log

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)

# pylint: enable= invalid-name

class Value(object):
    # pylint: disable= no-self-use
    """Class for all possible values in MinPy.

    It contains the real underlying value and the gradient information for auto differentiation.
    It also defines common operators and redirects the call to the namespace dispatcher.
    """
    _ns = None
    _ids = itertools.count(0)

    def __init__(self, context):
        self._bp_timestamp = -1
        self._minpy_value_id = next(self._ids)
        if context is None:
            self._context = current_context()
        else:
            self._context = context

    def is_marked_for_bp(self, tape):
        """Return whether the current `Value` will be used for autograd."""
        return tape != None and tape.is_recording and self._bp_timestamp == tape.timestamp

    def mark_for_bp(self, tape):
        """Set flag to record gradient information."""
        self._bp_timestamp = tape.timestamp

    def wait_to_read(self):
        """Wait for the internal data to be computed."""
        pass

    @property
    def context(self):
        """Return context of current `Value`."""
        return self._context

    @property
    def id(self): # pylint: disable= invalid-name
        """Return a unique id represents this value."""
        return self._minpy_value_id

    def __hash__(self):
        return id(self)

    def __cmp__(self, other):
        raise NotImplementedError()

    def __eq__(self, other):
        return Value._ns.equal(self, other)

    def __ne__(self, other):
        return Value._ns.not_equal(self, other)

    def __lt__(self, other):
        return Value._ns.less(self, other)

    def __gt__(self, other):
        return Value._ns.greater(self, other)

    def __le__(self, other):
        return Value._ns.less_equal(self, other)

    def __ge__(self, other):
        return Value._ns.greater_equal(self, other)

    def __pos__(self):
        raise NotImplementedError()

    def __neg__(self):
        return Value._ns.negative(self)

    def __abs__(self):
        return Value._ns.abs(self)

    def __invert__(self):
        raise NotImplementedError()

    def __round__(self, nbits):
        raise NotImplementedError()

    def __floor__(self):
        raise NotImplementedError()

    def __ceil__(self):
        raise NotImplementedError()

    def __trunc__(self):
        raise NotImplementedError()

    def __add__(self, other):
        return Value._ns.add(self, other)

    def __sub__(self, other):
        return Value._ns.subtract(self, other)

    def __mul__(self, other):
        return Value._ns.multiply(self, other)

    def __floordiv__(self, other):
        raise NotImplementedError()

    def __div__(self, other):
        return Value._ns.divide(self, other)

    def __truediv__(self, other):
        return Value._ns.true_divide(self, other)

    def __mod__(self, other):
        return Value._ns.mod(self, other)

    def __divmod__(self, other):
        raise NotImplementedError()

    def __pow__(self, other):
        return Value._ns.power(self, other)

    def __lshift__(self, other):
        raise NotImplementedError()

    def __rshift__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        raise NotImplementedError()

    def __xor__(self, other):
        raise NotImplementedError()

    def __radd__(self, other):
        return Value._ns.add(other, self)

    def __rsub__(self, other):
        return Value._ns.subtract(other, self)

    def __rmul__(self, other):
        return Value._ns.multiply(other, self)

    def __rfloordiv__(self, other):
        raise NotImplementedError()

    def __rdiv__(self, other):
        return Value._ns.divide(other, self)

    def __rtruediv__(self, other):
        return Value._ns.true_divide(other, self)

    def __rmod__(self, other):
        return Value._ns.mod(other, self)

    def __rdivmod__(self, other):
        return Value._ns.mod(other, self)

    def __rpow__(self, other):
        return Value._ns.power(other, self)

    def __rlshift__(self, other):
        raise NotImplementedError()

    def __rrshift__(self, other):
        raise NotImplementedError()

    def __rand__(self, other):
        raise NotImplementedError()

    def __ror__(self, other):
        raise NotImplementedError()

    def __rxor__(self, other):
        raise NotImplementedError()

    def __iadd__(self, other):
        return Value._ns.add(self, other)

    def __isub__(self, other):
        return Value._ns.subtract(self, other)

    def __imul__(self, other):
        return Value._ns.multiply(self, other)

    def __ifloordiv__(self, other):
        raise NotImplementedError()

    def __idiv__(self, other):
        return Value._ns.divide(self, other)

    def __itruediv__(self, other):
        return Value._ns.true_divide(self, other)

    def __imod__(self, other):
        return Value._ns.mod(self, other)

    def __ipow__(self, other):
        return Value._ns.power(self, other)

    def __ilshift__(self, other):
        raise NotImplementedError()

    def __irshift__(self, other):
        raise NotImplementedError()

    def __iand__(self, other):
        raise NotImplementedError()

    def __ior__(self, other):
        raise NotImplementedError()

    def __ixor__(self, other):
        raise NotImplementedError()
    # pylint: enable= no-self-use


class Number(Value, float):
    """Class for numbers with derivative information"""
    __slots__ = ['_val']

    def __new__(cls, val):
        return float.__new__(cls, val)

    def __init__(self, val, context=None):
        super(Number, self).__init__(context=context)
        self._val = val

    def __str__(self):
        return str(self._val)

    def __repr__(self):
        return repr(self._val)

    def get_data(self, dtype):
        """Get data of given type. Directly return the underlying value here."""
        return self._val

    def asnumpy(self):
        """ Get data in numpy compatible type """
        return self._val

    @property
    def val(self):
        """ return the underlying value """
        return self._val


class Array(Value):
    """Base array type.

    It provides convenient methods for arithmetic operations. The Array class
    is used for:
    1. Redirect all special member functions to corresponding pure function.
    2. Redirect normal member functions to correct member functions of
    underlying array object.
    """
    __slots__ = ['_data', '_latest_version']
    __array_priority__ = 100.0  # Highest priority when compute with numpy.ndarray.

    def __init__(self, data, context=None):
        super(Array, self).__init__(context)
        self._data = {}
        atype = Array.to_array_type(data)
        self._data[atype] = data
        self._latest_version = atype
        self._dtype = data.dtype

    @staticmethod
    def to_array_type(arr):
        """ Return the type enum of the given array """
        atype = type(arr)
        if atype == array_types['numpy']:
            return ArrayType.NUMPY
        elif atype == array_types['mxnet']:
            return ArrayType.MXNET
        else:
            raise TypeError('Array data of type {} unknown.'.format(atype))

    def __str__(self):
        return str(self.get_data(ArrayType.NUMPY))

    def __repr__(self):
        return repr(self.get_data(ArrayType.NUMPY))

    @property
    def context(self):
        return self._context

    @property
    def ndim(self):
        """ Number of array dimensions """
        # TODO(Yihe): add ndim in MXNet ndarray
        # return self._get_latest_data().ndim
        return self.get_data(ArrayType.NUMPY).ndim

    def has_type(self, atype):
        """ Return whether array data of given type exists in the underlying storage.
        """
        return atype in self._data.keys()

    def reshape(self, *args, **kwargs):
        """Function for reshape this array.

        Usage example:

        ::

            a = np.ones([10, 10])
            b = a.reshape([5, 20])
            b = a.reshape(5, 20)

        See `here <http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html>`_
        for further explanation.

        Parameters
        ----------
        args
            A single iterable or a sequence of integers representing a
            new shape.  Although it being an iterable is not
            documented in official document, it is renowned and widely
            used in practice.

        Returns
        -------
        Array
            Reshaped array.
        """
        if len(args) == 1 and isinstance(args[0], collections.Iterable):
            new_shape = args[0]
        else:
            new_shape = tuple(x for x in args)
        if 'order' in kwargs and kwargs['order'] != 'C':
            raise NotImplementedError(
                'Orders other than C are not currently supported.')
        return Value._ns.reshape(self, new_shape)

    def dot(self, other, out=None):
        """Function for dot production. """
        if out is not None:
            # TODO: Support out argument
            raise ValueError('Out option is not supported.')
        return Value._ns.dot(self, other)

    def argmax(self, axis=None, out=None):
        """Returns the indices of the maximum values along an axis

        Parameters
        ----------
        axis
            By default, the index is into the flattened array,
            otherwise along the specified axis.
        out
            If provided, the result will be inserted into this array.
            It should be of the appropriate shape and dtype.

        Returns
        -------
        Array
            Array of indices into the array.
        """
        if out is not None:
            # TODO: Support out argument
            raise ValueError('Out option is not supported.')
        return Value._ns.argmax(self, axis)

    def _synchronize_data(self):
        """Synchronize the data of different array types. """
        if self._latest_version == ArrayType.MXNET:
            _logger.info(
                'Copy from MXNet array to NumPy array for Array "{}" of shape {}.'
                .format(id(self), self.shape))
            mxarray = self._data[ArrayType.MXNET]
            self._data[ArrayType.NUMPY] = mxarray.asnumpy()
        elif self._latest_version == ArrayType.NUMPY:
            _logger.info(
                'Copy from NumPy array to MXNet array for Array "{}" of shape {}.'
                .format(id(self), self.shape))
            nparray = self._data[ArrayType.NUMPY]
            self._data[ArrayType.MXNET] = mxnet.ndarray.array(
                nparray, ctx=self._context.as_mxnet_context())
        self._latest_version = None

    def enforce_data(self, dtype):
        """Enforce array data of given type."""
        if self._latest_version is not None and self._latest_version != dtype:
            self._synchronize_data()

    def get_data(self, dtype):
        """Get array data of given type."""
        self.enforce_data(dtype)
        return self._data[dtype]

    @property
    def dtype(self):
        """Return contained dtype, in NumPy's dtype object"""
        return self._dtype

    def _get_latest_data(self):
        """Return the latest version of the raw data"""
        if self._latest_version is not None:
            return self._data[self._latest_version]
        else:
            if self.has_type(ArrayType.NUMPY):
                return self._data[ArrayType.NUMPY]
            else:
                return self._data[ArrayType.MXNET]

    def asnumpy(self):
        """Get raw NumPy array.

        This will return a copied array of numpy.ndarray type
        """
        return numpy.array(self.get_data(ArrayType.NUMPY))

    def get_data_mutable(self, dtype):
        """Get exclusive access to array data of given type."""
        if self._latest_version is not None and self._latest_version != dtype:
            self._synchronize_data()
        self._latest_version = dtype
        return self._data[dtype]

    @property
    def shape(self):
        """Get the shape of array."""
        return self._get_latest_data().shape

    def __getitem__(self, index):
        """NumPy indexing operations.

        Currently `mxnet.ndarray` does not support full indexing, so there is an implicit
        conversion to NumPy array.
        """
        np_index = None
        to_np = lambda x: x if isinstance(x, slice) else wrap(x).get_data(ArrayType.NUMPY)
        if isinstance(index, tuple):
            np_index = tuple(to_np(i) for i in index)
        else:
            np_index = to_np(index)
        return Value._ns._minpy_getitem(self, np_index)

    def __setitem__(self, index, val):
        """NumPy indexing operations.

        Currently `mxnet.ndarray` does not support full indexing, so there is an implicit
        conversion to NumPy array. Also note that this operation breaks gradient chain.
        """
        np_index = None
        np_val = wrap(val).get_data(ArrayType.NUMPY)
        to_np = lambda x: x if isinstance(x, slice) else wrap(x).get_data(ArrayType.NUMPY)
        if isinstance(index, tuple):
            np_index = tuple(to_np(i) for i in index)
        else:
            np_index = to_np(index)
        np_array = self.get_data_mutable(ArrayType.NUMPY)
        np_array.__setitem__(np_index, np_val)

    def __delitem__(self, index):
        """NumPy indexing operations.

        Currently `mxnet.ndarray` does not support full indexing, so there is an implicit
        conversion to NumPy array.  Also note that this operation breaks gradient chain.
        """
        self.get_data_mutable(ArrayType.NUMPY).__delitem(index)

    # pylint: disable= invalid-name
    @property
    def T(self):
        """Get transposed array."""
        return Value._ns.transpose(self)
    # pylint: enable= invalid-name

    @property
    def size(self):
        """Get number of elements in the array."""
        return self._get_latest_data().size

    def wait_to_read(self):
        """Wait until the internal data has been calculated.

        If the array only contains numpy data, it will simply return. Otherwise,
        it will wait until the mxnet data is finished.
        """
        if self.has_type(ArrayType.MXNET):
            self.get_data(ArrayType.MXNET).wait_to_read()

def _make_wrapper_types():
    """Create dictionary from underlying data type to its wrapper type.

    For types that have no corresponding wrapper type, just return the input type.
    """
    ret = collections.defaultdict(lambda: lambda x: x)
    for i in array_types.values():
        ret[i] = Array
    for ty_list in number_types.values():
        for i in ty_list:
            ret[i] = Number
    return ret

_wrapper_types = _make_wrapper_types()  # pylint: disable= invalid-name

def wrap(data):
    """Wrap given data into its corresponding wrapper class.

    For example, :class:`numpy.ndarray` will be converted to
    :class:`Array` while float number will become
    :class:`Number`. The allowed array types are defined in
    :class:`minpy.array_variants.array_types`; the allowed number
    types are defined in
    :class:`minpy.array_variants.number_types`.
    """
    if data is None:
        return None
    if hasattr(data, '_minpy_value_id'):
        return data
    else:
        dtype = type(data)
        return _wrapper_types[dtype](data)

    '''
    elif isinstance(data, list):
        return [wrap(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(wrap(d) for d in data)
    elif isinstance(data, dict):
        return {k: wrap(v) for k, v in data.items()}
    else:
        raise TypeError('Cannot wrap type of "{}".'.format(dtype))
    '''
