'''Provides ops for different kind of visualization with TensorBoard.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import minpy

from minpy.visualize.summary_pb2 import Summary

def scalar_summary(tag, scalar):
    '''Op for scalar visualization.

	Args:
	- tag: A string. Name of a summary like 'loss'.
	- scalar: A 'float', 'int', 'long',
	'minpy.array.Array' with only one element,
	or 'numpy.ndarray' with only one element."

	Returns:
	- A `Summary` protocol buffer with scalar summary.
	'''
    if isinstance(scalar, minpy.array.Array):
        scalar = float(scalar.asnumpy()[0])
    elif isinstance(scalar, numpy.ndarray):
        scalar = float(scalar[0])
    return Summary(value=[Summary.Value(tag=tag, simple_value=scalar)])
