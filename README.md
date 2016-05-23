# MinPy

[![Build Status](https://travis-ci.org/dmlc/minpy.svg?branch=master)](https://travis-ci.org/dmlc/minpy)

This repository aims at prototyping a pure `numpy` interface above [mxnet](https://github.com/dmlc/mxnet) backend. The key features include:

* [Autograd](https://github.com/HIPS/autograd) support.
* Nature MXNet symbol integration.
* Graceful fallback for missing operations.
* Transparent device and partition specification.

How to get started?
-------------------
The project is still a work-in-progress. You could look at this [tutorial](https://github.com/dmlc/minpy/blob/master/examples/demo/minpy_tutorial.ipynb) to understand its concept. Documents are coming soon!

# Installation guide for development

Currently MinPy is going through rapid development (but we do our best to keep the API unchanged). So it is adviced to do an editable installation.
Change directory into where the Python package lies and run `pip install -e .` if you are in a virtual environment, or `pip install --user -e .` if you are using your system Python packages. This will ensure a symbolic link to the project, so you do not have to install a second time when you update this repository.

What we really want?
-------------------
In one word, if you have a `numpy` code, you could replace the `import` by:
```python
import minpy.numpy as np

# other numpy codes remain the same
```

and you could have:
* Auto differentiation support.
* Speed up with some operations executed on GPUs.
* Missing operations will not cause "NO IMPLEMENTATION" exception.
* Directly call Caffe's Layer abstraction without any code change.
* Switch between `numpy`'s operators and Caffe's operator as you wish.
