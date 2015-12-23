from __future__ import absolute_import

from mxnet import random
from . import ndarray_wrapper

ndarray_wrapper.wrap_namespace(random.__dict__, globals())
