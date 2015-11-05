# Minpy

This repository aims at prototyping a pure `numpy` interface above [mxnet](https://github.com/dmlc/mxnet) backend. The key features include:

* [Autograd](https://github.com/HIPS/autograd) support.
* Nature [Caffe](https://github.com/BVLC/caffe) layer integration.
* Graceful fallback for missing operations.
* Transparent device and partition specification.

What we really want?
-------------------
In one word, if you have a `numpy` code, you could replace the `import` by:
```python
import minpy as np

# other numpy codes remain the same
```

and you could have:
* Auto differentiation support.
* Speed up with some operations executed on GPUs.
* Missing operations will not cause "NO IMPLEMENTATION" exception.
* Directly call Caffe's Layer abstraction without any code change.
* Switch between `numpy`'s operators and Caffe's operator as you wish.
