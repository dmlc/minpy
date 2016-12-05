# MinPy

[![Build Status](https://travis-ci.org/dmlc/minpy.svg?branch=master)](https://travis-ci.org/dmlc/minpy)
[![PyPI version](https://badge.fury.io/py/minpy.svg)](https://badge.fury.io/py/minpy)
[![Docs](https://readthedocs.org/projects/minpy/badge/?version=latest)](https://minpy.readthedocs.io/en/latest/)

This repository aims at prototyping a pure [NumPy](http://www.numpy.org/) interface above [MXNet](https://github.com/dmlc/mxnet) backend. The key features include:

* [Autograd](https://github.com/HIPS/autograd) support. Automatic gradient generation.
* Seamless MXNet symbol integration.
* Graceful fallback for missing operations to NumPy.
* Transparent device and partition specification.

## MXNet version

Currently both MXNet and MinPy are going through rapid development. MinPy is not guaranteed to work with all MXNet versions.

This version of MinPy is tested to work with MXNet at b7ab768 on `nnvm` branch.

## How to get started

The project is still a work-in-progress. You could look at this [tutorial](https://github.com/dmlc/minpy/blob/master/examples/demo/minpy_tutorial.ipynb) to understand its concept. Documents are hosted [here](https://minpy.readthedocs.io/en/latest/).

## Easy installation

```
pip install minpy
```

## What we could achieve

In one word, if you have NumPy code, you could replace the `import` by:

```python
import minpy.numpy as np
```

Other numpy code remain the same. And you could have:
* Auto differentiation support.
* Speed up with some operations executed on GPUs.
* Missing operations will fall back to NumPy version automatically.
* Hybrid programming with efficient MXNet's symbol and flexible MinPy arrays.
