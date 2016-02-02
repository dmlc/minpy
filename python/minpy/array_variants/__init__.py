from ..utils import common

variants=['numpy']

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
