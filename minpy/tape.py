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
            self._result_grad_records[result].append(grad_rec)
        else:
            for sub_result in result:
                self._result_grad_records[result].append(grad_rec)

    def _set_gradient_target(self, target):
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

    def get_gradient(self, origin, target):
        """Get gradient of the specified array.

        This will first set the gradients of target (using value 1.0) and
        then compute the gradient using the backward path recorded during
        forward computation. Currently, we use BFS order if more than one
        gradient operators could be computed at the same time.

        Parameters
        ----------
        origin
            A tuple of array objects representing the inputs.

        target 
            Array or a tuple of array representing the target.

        Returns
        -------
        tuple of Array
            The gradient of input arrays.
        """
        def remove_grad_record(owner, grad_record):
            """Remove gradient record from owner.

            Return true if the owner has no other grad records to be resolved
            (and thus has finished gradient computation).
            """
            assert isinstance(owner, array.Value)
            # TODO(minjie): could be optimized using pop.
            self._array_grad_records[owner].remove(grad_record)
            if len(self._array_grad_records[owner]) == 0:
                self._array_grad_records.pop(owner, None)
                return True
            else:
                return False

        def compute_grad_record(grad_record):
            """Run the function in the grad record."""
            if isinstance(grad_record.result, array.Value):
                return grad_record.grad_func(self._grads[grad_record.result])
            else:
                return grad_record.grad_func(
                        tuple(self._grads[rst] for rst in grad_record.result))

        origin_id = tuple(id(arr) for arr in origin)

        # Set gradient target.
        self._set_gradient_target(target)

        # Initialize bfs queue.
        bfs_queue = collections.deque()
        if isinstance(target, array.Value):
            bfs_queue.append(target)
        else:
            for sub_target in target:
                bfs_queue.append(sub_target)

        # TODO(minjie): require an extra traversal before this to eliminate
        # those grad_records that do not contribute to the target.
        # Example:
        # def foo(x):
        #   y = x + 1
        #   z = print(np.argmax(x))
        #   return y
        # The argmax above should not be involed in gradient computation.
        
        while len(bfs_queue) != 0:
            #print(len(self._grads), len(self._array_grad_records), len(self._result_grad_records))
            current = bfs_queue.popleft()
            # Resolve all grad_records that will use gradient of the current array.
            grad_records = self._result_grad_records[current]
            while len(grad_records) != 0:
                grad_record = grad_records.pop()
                # TODO(minjie): add primitive_type info in debug info later.
                _logger.debug(
                    'Calling derivative func "%s"', grad_record.grad_func)
                # TODO(minjie): this may raise error if the grad_record has multiple
                # results. We should check the all the result gradients are available
                # before calling this function.
                grad = compute_grad_record(grad_record)
                owner = grad_record.owner
                self._cumulate_gradient(owner, grad)
                # Remove grad_record in forward path and trigger those arrays
                # that all their grad_records in the forward path have been
                # resolved.
                if isinstance(owner, array.Value):
                    if remove_grad_record(owner, grad_record):
                        bfs_queue.append(owner)
                elif owner is not None:
                    for sub_owner in owner:
                        if remove_grad_record(sub_owner, grad_record):
                            bfs_queue.append(sub_owner)
            # Release memory of the gradient of the current array.
            self._result_grad_records.pop(current, None)
            if not id(current) in origin_id:
                self._grads.pop(current, None)

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
