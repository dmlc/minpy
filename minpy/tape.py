#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tape for recording gradient calculation."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
import contextlib
import collections

import numpy
import mxnet

from . import array
from .utils import log

# pylint: disable=invalid-name
_logger = log.get_logger(__name__)
# pylint: enable=invalid-name

GradRecord = collections.namedtuple(
    'GradRecord', ['grad_func', 'result', 'owner'])


class Tape(object):
    """Records gradient calculation."""
    global_tape = None
    timestamp = 0

    def __init__(self):
        # Stores grad value result from target back to [KEY]. Array -> grad result (Array)
        #self._grads = collections.defaultdict(lambda: array.wrap(0.0))
        self._grads = {}
        self._grads_refcount = {}
        # Store derivation path (forward). Array -> list of grad records
        # This maps from arrays to the gradient functions that compute their gradient.
        self._array_grad_records = collections.defaultdict(list)
        # Store derivation path (backward). Array -> list of grad records
        # This maps from arrays to the gradient functions that use them as inputs.
        self._result_grad_records = collections.defaultdict(list)

        self.__class__.timestamp += 1

    def add_partial_derivative(self, grad_func, owner, result):
        """Add partial derivative.

        Parameters
        ----------
        grad_func
            Function for calculating gradient.
        owner
            Owners of the gradient. Usually it is just one array. But a list is also allowed.
        result
            Result of previous forward execution.
        """
        grad_rec = GradRecord(grad_func=grad_func, result=result, owner=owner)
        # Create forward derivation path.
        if isinstance(owner, array.Value):
            self._array_grad_records[owner].append(grad_rec)
        elif owner is not None:
            for sub_owner in owner:
                self._array_grad_records[sub_owner].append(grad_rec)
        else:
            # None means a placeholder for an array that needs no gradient.
            pass
        # Create backward derivation path.
        if isinstance(result, array.Value):
            if not result in self._grads_refcount:
                self._grads_refcount[result] = 0
            self._grads_refcount[result] += 1
            self._result_grad_records[result].append(grad_rec)
        else:

    def set_gradient_target(self, target):
        """Set gradient targets to ones."""
        # Set gradient target for one.
        if isinstance(target, array.Value):
            self._grads[target] = array.wrap(1.0 if isinstance(
                target, array.Number) else numpy.ones(target.shape))
        else:
            for sub_target in target:
                self.set_gradient_target(sub_target)

    def _is_gradable(self, current_array):
        """Check if the gradient can now be calculated relative to the specified array.

        It means that all resulting arrays taken the specified array as input
        have gradients calculated.
        """
        if isinstance(current_array, array.Value):
            return len(self._array_grad_records[current_array]) == 0
        else:
            for arr in current_array:
                if len(self._array_grad_records[arr]) != 0:
                    return False
            return True

    def _cumulate_gradient(self, arr, grad):
        """Cumulate gradients belonging to the same array.

        Recurse when handle multiple arrays.
        """
        #TODO(minjie): If arr is a list of arrays, redundant computation may happen.
        if isinstance(arr, array.Value):
            current_gradient = array.wrap(grad)
            if arr in self._grads:
                self._grads[arr] += current_gradient
            else:
                self._grads[arr] = current_gradient
        elif arr is not None:
            if len(arr) != len(grad):
                _logger.fatal('Number of gradients does not match.')
            for sub_arr, sub_grad in zip(arr, grad):
                self._cumulate_gradient(sub_arr, sub_grad)

    def get_gradient(self, origin):
        """Get gradient of the specified array.

        Parameters
        ----------
        origin
            A tuple of array objects that require gradient computation.

        Returns
        -------
        tuple of Array
            The gradient of origin arrays.

        Target gradients must be set prior to this function call.
        """
        # Initialize the dfs stack.
        dfs_stack = []

        def push_grad_record(grad_record):
            """Push gradient record into stack."""
            dfs_stack.append(grad_record)

        def remove_grad_record(owner, grad_record):
            """Remove gradient record from owner."""
            if isinstance(owner, array.Value):
                self._array_grad_records[owner].remove(grad_record)
                if len(self._array_grad_records[owner]) == 0:
                    self._array_grad_records.pop(owner, None)
            elif owner is not None:
                for sub_owner in owner:
                    remove_grad_record(sub_owner, grad_record)

        for arr in origin:
            assert isinstance(arr, array.Value), 'Type is not `array.Value`.'
            for grad_record in self._array_grad_records[arr]:
                if grad_record.result in self._grads:
                    # The record could be computed.
                    grad = grad_record.grad_func(self._grads[grad_record.result])
                    self._cumulate_gradient(grad_record.owner, grad)
                else:
                    push_grad_record(grad_record)

        while len(dfs_stack) != 0:
            current_record = dfs_stack[-1]
            result_array = current_record.result
            # TODO(minjie): should not use this, should use _gradable instead
            if result_array in self._grads:
                # The record could be computed.
                dfs_stack.pop()
                # TODO(minjie): add primitive_type info in debug info later.
                _logger.debug(
                    'Calling derivative func "%s"', current_record.grad_func)
                grad = current_record.grad_func(self._grads[result_array])
                self._cumulate_gradient(current_record.owner, grad)
                # Also remove the gradient record from the owner.
                remove_grad_record(current_record.owner, current_record)
                # Check reference count of result array.
                self._grads_refcount[result_array] -= 1
                rc = self._grads_refcount[result_array]
                if rc == 0:
                    print('delete', len(self._grads), id(grad_record.result))
                    self._grads.pop(result_array, None)
                    self._grads_refcount.pop(result_array, None)
                elif rc < 0:
                    _logger.fatal('Invalid reference count for gradient arrays')
            else:
                assert len(self._array_grad_records[result_array]) != 0
                # Require computing the result of this record.
                for rec in self._array_grad_records[result_array]:
                    push_grad_record(rec)

        # TODO(minjie): If the gradient cannot be computed, this should return a zero array
        # of proper shape.
        return tuple(self._grads[arr] for arr in origin)


@contextlib.contextmanager
def tape():
    """Convenience context wrapper for creating temporary `Tape`."""
    Tape.global_tape = Tape()
    yield Tape.global_tape
    Tape.global_tape = None


def global_tape():
    """Returns current global `Tape`."""
    return Tape.global_tape
