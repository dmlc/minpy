from __future__ import absolute_import

import types
import functools
import numpy as np

from ..core import Primitive

def unbox_args(f):
    return functools.wraps(f)(lambda *args, **kwargs: f(*args, **kwargs))

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {np.int, np.int8, np.int16, np.int32, np.int64, np.integer}
    function_types = {np.ufunc, types.FunctionType, types.BuiltinFunctionType}

    for name, obj in old.items():
        if type(obj) in function_types:
            new[name] = Primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = obj
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(np.__dict__, globals())
