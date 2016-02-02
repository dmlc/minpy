from ..utils import common

class FunctionType(common.AutoNumber):
    """Enumeration of types of functions.

    Semantically this is different from :class:`..array.ArrayType`,
    but for now one data type corresponds to one function type.
    """
    NUMPY = ()
    MXNET = ()

class ArrayType(common.AutoNumber):
    """Enumeration of types of arrays."""
    NUMPY = ()
    MXNET = ()

variants = {
        'numpy': (ArrayType.NUMPY, FunctionType.NUMPY),
        #'mxnet': (ArrayType.MXNET, FunctionType.MXNET)
        }
