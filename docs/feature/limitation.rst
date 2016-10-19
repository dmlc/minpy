Limitation of MinPy
===================

* Autograd system does not support in-place array operations, such as ``A[3] = 3``.
* Since NumPy has many submodules, not all submodules are currently supported.
* Some objects lacks in ``minpy.numpy`` namespace, such as type objects, like ``minpy.numpy.float32``.