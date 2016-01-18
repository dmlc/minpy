from __future__ import absolute_import

import types
import functools
import mxnet.ndarray as nd

from ..core import Primitive

def unbox_args(f):
    return functools.wraps(f)(lambda *args, **kwargs: f(*args, **kwargs))

def wrap_namespace(old, new):
    function_types = {types.FunctionType, types.BuiltinFunctionType}
    for name, obj in old.items():
        if type(obj) in function_types:
            new[name] = Primitive(obj)
        else:
            new[name] = obj

wrap_namespace(nd.__dict__, globals())
