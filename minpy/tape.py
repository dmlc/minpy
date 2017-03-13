#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tape for recording gradient calculation."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement

import contextlib
import collections
import copy

import numpy

from . import array
from .utils import log

# pylint: disable=invalid-name
_logger = log.get_logger(__name__)
# pylint: enable=invalid-name

GradRecord = collections.namedtuple('GradRecord',
                                    ['grad_func', 'result', 'owner'])


class Tape(object):
    """Records gradient calculation."""
    global_tape = None
    timestamp = 0

    def __init__(self):
        # Stores grad value result from target back to [KEY]. Array -> grad result (Array)
        self._grads = {}
        # Store derivation path (forward). ArrayId -> reference counts
        # This maps from arrays to the number of gradient functions required to compute its
        # gradient.
        self._array_grad_refcount = collections.defaultdict(int)
        # Store derivation path (backward). ArrayId -> list of grad records
        # This maps from arrays to the gradient functions that use them as inputs.
        self._result_grad_records = collections.defaultdict(list)
        self._recording = False
        self.__class__.timestamp += 1

    def start_recording(self):
        """Start recording gradient path for each primitive called afterwards."""
        self._recording = True

    def stop_recording(self):
        """Stop recording gradient path for each primitive called afterwards."""
        self._recording = False

    @property
    def is_recording(self):
        """Return whether the tape is recording gradient path."""
        return self._recording

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
        if not self._recording:
            return
        grad_rec = GradRecord(grad_func=grad_func, result=result, owner=owner)
        # Create forward derivation path.
        if isinstance(owner, array.Value):
            #print('add', owner.id)
            self._array_grad_refcount[owner.id] += 1
        elif owner is not None: # None means a placeholder for an array that needs no gradient.
            for sub_owner in owner:
                if isinstance(sub_owner, array.Value):
                    self._array_grad_refcount[sub_owner.id] += 1
        # Create backward derivation path.
        if isinstance(result, array.Value):
            self._result_grad_records[result.id].append(grad_rec)
        else:
            for sub_result in result:
                self._result_grad_records[sub_result.id].append(grad_rec)

    def _set_gradient_target(self, target):
        """Set gradient targets to ones."""
        # Set gradient target for one.
        if isinstance(target, array.Value):
            self._grads[target.id] = array.wrap(1.0 if isinstance(
                target, array.Number) else numpy.ones(target.shape))
        else:
            for sub_target in target:
                self._set_gradient_target(sub_target)

    def _cumulate_gradient(self, arr, grad):
        """Cumulate gradients belonging to the same array.

        Recurse when handle multiple arrays.
        """
        if isinstance(arr, array.Value):
            current_gradient = array.wrap(grad)
            if arr.id in self._grads:
                self._grads[arr.id] += current_gradient
            else:
                self._grads[arr.id] = current_gradient
        elif arr is not None:
            if len(arr) != len(grad):
                _logger.fatal('Number of gradients does not match.')
            for sub_arr, sub_grad in zip(arr, grad):
                self._cumulate_gradient(sub_arr, sub_grad)

    def _prune_gradient_path(self, target_queue):
        """Remove gradients that are not required to be computed.

        Traversal from target to eliminate grad_records that do not
        contribute to the target.
        Example:
        def foo(x):
          y = x + 1
          z = print(np.argmax(y))
          return y
        The argmax above should not be involed in gradient computation.
        """
        # pylint: disable= too-many-nested-blocks, too-many-branches
        touched = set()
        touch_queue = copy.deepcopy(target_queue)
        while len(touch_queue) != 0:
            current_id = touch_queue.popleft()
            if current_id in touched:
                continue
            else:
                touched.add(current_id)
            for grad_record in self._result_grad_records[current_id]:
                owner = grad_record.owner
                if isinstance(owner, array.Value):
                    touch_queue.append(owner.id)
                else:
                    for sub_owner in owner:
                        if isinstance(sub_owner, array.Value):
                            touch_queue.append(sub_owner.id)
        to_delete = []
        for arrid, grad_records in self._result_grad_records.items():
            if not arrid in touched:
                for grad_record in grad_records:
                    owner = grad_record.owner
                    if isinstance(owner, array.Value):
                        self._array_grad_refcount[owner.id] -= 1
                    else:
                        for sub_owner in owner:
                            if isinstance(sub_owner, array.Value):
                                self._array_grad_refcount[sub_owner.id] -= 1
                to_delete.append(arrid)
        for arrid in to_delete:
            self._result_grad_records.pop(arrid, None)
        # pylint: enable= too-many-nested-blocks, too-many-branches

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
        # pylint: disable= too-many-locals, too-many-branches
        def decr_refcount(owner):
            """Decrement the reference count of the given owner.

            Return true if the owner has no other grad records to be resolved
            (and thus has finished gradient computation).
            """
            assert isinstance(owner, array.Value)
            assert owner.id in self._array_grad_refcount, owner.id
            self._array_grad_refcount[owner.id] -= 1
            if self._array_grad_refcount[owner.id] == 0:
                self._array_grad_refcount.pop(owner.id, None)
                return True
            else:
                return False

        def compute_grad_record(grad_record):
            """Run the function in the grad record."""
            if isinstance(grad_record.result, array.Value):
                return grad_record.grad_func(self._grads[grad_record.result.id])
            else:
                return grad_record.grad_func(
                    tuple(self._grads[rst.id] for rst in grad_record.result))

        origin_id = set(arr.id for arr in origin)

        # Set gradient target.
        self._set_gradient_target(target)

        # Initialize bfs queue.
        bfs_queue = collections.deque()
        if isinstance(target, array.Value):
            bfs_queue.append(target.id)
        else:
            for sub_target in target:
                bfs_queue.append(sub_target.id)

        # First do an extra traversal to remove extra gradients.
        self._prune_gradient_path(bfs_queue)

        # Compute gradients from target to origin.
        while len(bfs_queue) != 0:
            current_id = bfs_queue.popleft()
            # Resolve all grad_records that will use gradient of the current array.
            grad_records = self._result_grad_records[current_id]
            while len(grad_records) != 0:
                grad_record = grad_records.pop()
                # TODO(minjie): add primitive_type info in debug info later.
                _logger.debug(
                    'Calling derivative func "%s"', grad_record.grad_func)
                # TODO(minjie): this may raise error if the grad_record has multiple
                # results. We should check all the result gradients are available
                # before calling this function.
                grad = compute_grad_record(grad_record)
                owner = grad_record.owner
                self._cumulate_gradient(owner, grad)
                # Remove grad_record in forward path and trigger those arrays
                # whose grad_records in the forward path have been resolved.
                if isinstance(owner, array.Value):
                    if decr_refcount(owner):
                        bfs_queue.append(owner.id)
                elif owner is not None:
                    for sub_owner in owner:
                        if isinstance(sub_owner, array.Value) and decr_refcount(sub_owner):
                            bfs_queue.append(sub_owner.id)
            # Release memory of the gradient of the current array.
            self._result_grad_records.pop(current_id, None)
            if not current_id in origin_id:
                self._grads.pop(current_id, None)

        origin_grad = []
        for arr in origin:
            if arr.id in self._grads:
                origin_grad.append(self._grads[arr.id])
            else:
                # The gradient of this array is zero.
                # TODO(minjie): This may need to return a zero array of proper shape.
                origin_grad.append(0.0)
        return origin_grad
        # pylint: enable= too-many-locals, too-many-branches


@contextlib.contextmanager
def tape():
    """Convenience context wrapper for creating temporary `Tape`."""
    Tape.global_tape = Tape()
    yield Tape.global_tape
    Tape.global_tape = None


def global_tape():
    """Returns current global `Tape`."""
    return Tape.global_tape
