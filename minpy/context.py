#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=logging-format-interpolation
"""Context management API of minpy."""
from __future__ import absolute_import

import mxnet

from .utils import log

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)

# pylint: enable= invalid-name


class Context(object):
    """Constructing a context.

    Parameters
    ----------
    device_type : {'cpu', 'gpu'} or Context.
        String representing the device type.

    device_id : int (default=0)
        The device id of the device, needed for GPU.

    Note
    ----
    Context can also be used a way to change default context.

    Examples
    --------
    >>> import minpy as np
    >>> # array on cpu
    >>> cpu_array = np.ones((2, 3))
    >>> # switch default context to GPU(2)
    >>> with np.Context(np.gpu(2)):
    >>>     gpu_array = np.ones((2, 3))
    >>> gpu_array.context
    gpu(2)
    """
    # static class variable
    default_ctx = None
    devtype2str = {1: 'cpu', 2: 'gpu', 3: 'cpu_pinned'}
    devstr2type = {v: k for k, v in devtype2str.items()}

    def __init__(self, device_type, device_id=0):
        if isinstance(device_type, Context):
            self.device_typeid = device_type.device_typeid
            self.device_id = device_type.device_id
        else:
            self.device_typeid = Context.devstr2type[device_type]
            self.device_id = device_id
        self._mxnet_context = mxnet.Context(self.devtype2str[self.device_typeid],
                                            self.device_id)
        self._old_ctx = None

    @property
    def device_type(self):
        """Return device type of current context.

        Returns
        -------
        device_type : str
        """
        return Context.devtype2str[self.device_typeid]

    def __eq__(self, other):
        """Compare two contexts. Two contexts are equal if they
        have the same device type and device id.
        """
        if not isinstance(other, Context):
            return False
        if self.device_typeid == other.device_typeid and \
                self.device_id == other.device_id:
            return True
        return False

    def as_mxnet_context(self):
        """Get MXNet context."""
        _logger.debug('Getting MXNet context with typeid "{}" and id "{}"'.
                      format(self.device_typeid, self.device_id))
        return self._mxnet_context

    def __str__(self):
        return '%s(%d)' % (self.device_type, self.device_id)

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        self._old_ctx = Context.default_ctx
        Context.default_ctx = self
        return self

    def __exit__(self, ptype, value, trace):
        Context.default_ctx = self._old_ctx

# initialize the default context in Context
Context.default_ctx = Context('cpu', 0)


def cpu(device_id=0):
    """Return a CPU context.

    This function is a short cut for Context('cpu', device_id).

    Parameters
    ----------
    device_id : int, optional
        The device id of the device. device_id is not needed for CPU.
        This is included to make interface compatible with GPU.

    Returns
    -------
    context : Context
        The corresponding CPU context.
    """
    return Context('cpu', device_id)


def gpu(device_id=0):
    """Return a GPU context.

    This function is a short cut for Context('gpu', device_id).

    Parameters
    ----------
    device_id : int, optional
        The device id of the device, needed for GPU.

    Returns
    -------
    context : Context
        The corresponding GPU context.
    """
    return Context('gpu', device_id)


def current_context():
    """Return the current context.

    Returns
    -------
    default_ctx : Context
    """
    return Context.default_ctx


def set_context(ctx):
    """Set current context."""
    Context.default_ctx = ctx
