import numpy as _np
import mxnet.ndarray as _nd
import mxnet.context as _context


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


def unpack_batch(batch):
    """
    Inputs:
      - batch: a batch provided by mxnet.io.DataIter
    Returns:
      - data: mx.ndarray.NDArray
      - labels: mx.ndarray.NDArray
    """
    context = _context.Context.default_ctx
    return batch.data[0].as_in_context(context), batch.label[0].as_in_context(context)


def count_parameters(params):
    """
    Inputs:
      - params: a list, tuple or dict of arrays (mxnet.ndarray.NDArray)
    Returns:
      - n: total number of parameters
    """
    if isinstance(params, dict): params = params.values()

    assert all(isinstance(param, _nd.NDArray) for param in params), \
        'All parameters must be mxnet.ndarray.NDArray.'

    return sum(param.size for param in params)
