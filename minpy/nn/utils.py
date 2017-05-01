import numpy as _np
import mxnet.ndarray as _nd

def _np_decorator(f):
    def wrapped(*args, **kwargs):
        convert = lambda array : array.asnumpy() if isinstance(array, _nd.NDArray) else array
        args = tuple(map(convert, args))
        kwargs = dict(zip(kwargs.keys(), tuple(map(convert, tuple(kwargs.values())))))
        return f(*args, **kwargs)

    return wrapped
        
@_np_decorator
def cross_entropy(p, labels):
    labels = labels.astype(_np.int)
    return -_np.mean(_np.log(p[_np.arange(len(p)), labels]))
