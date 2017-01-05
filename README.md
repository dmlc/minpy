# MinPy

[![Build Status](https://travis-ci.org/dmlc/minpy.svg?branch=master)](https://travis-ci.org/dmlc/minpy)
[![PyPI version](https://badge.fury.io/py/minpy.svg)](https://badge.fury.io/py/minpy)
[![Docs](https://readthedocs.org/projects/minpy/badge/?version=latest)](https://minpy.readthedocs.io/en/latest/)

This repository aims at prototyping a pure [NumPy](http://www.numpy.org/) interface above [MXNet](https://github.com/dmlc/mxnet) backend. In one word, if you have NumPy code, with
```python
import minpy.numpy as np
```
* Operators with GPU support will be ran on GPU automatically.
* Graceful fallback for missing operations to NumPy.
* [Autograd](https://github.com/HIPS/autograd) support. Automatic gradient generation.
* Seamless MXNet symbol integration.

## Why NumPy interface?

Why obsessed with NumPy interface? First of all, programming in NumPy is very nature and has been favored by academic researches for many years. To have a taste of its flexibility, let's look at some examples below.

### Printing and Debugging
(code snipplet)

### Data-dependent branches
(code snipplet)

### Dynamic shapes
(code snipplet)


Tensorflow is just one typical example, many other packages (e.g. Theano, or even MXNet) have similar problems. The underlying reason is the trade-off between *symbolic programming* and *imperative programming*. Codes in symbolic programs (Tensorflow and Theano) are *not* doing concrete computation but generating dataflow graph instead. This enables extensive optimizations but requires reinventing almost all language constructs (like if and loop). Imperative programs (NumPy) generates dataflow graph *along with* the computation, so you could freely query or use the value just computed. For a rough analogy, symbolic programming is like C++, where you have two stages: compilation (generating dataflow graph) and execution (concrete evaluation), while imperative programming is like Python -- it is easy to tweak but less efficient.

In MinPy, we use NumPy syntax to ease your programming, while achieving good performance at the same time.

## Is MinPy fast?
The imperative interface does raise many challenges, but MinPy is managed to achieve good speed when the workload is not too tiny.

(MLP performance chart)

(RNN performance chart)


## Get Started
### MXNet version

Currently both MXNet and MinPy are going through rapid development. MinPy is not guaranteed to work with all MXNet versions.

This version of MinPy is tested to work with MXNet on `engine` branch. So use following command when you download and build MXNet from source:
```
git clone --recursive -b engine https://github.com/dmlc/mxnet.git
```
Please read [installation guide](https://minpy.readthedocs.io/en/latest/get-started/install.html) for more details.

### NumPy version

Minpy prototypes a pure Numpy interface. To make the interface consistent, please make sure Numpy version >= 1.10.0 before install Minpy.

### Easy installation

```
pip install minpy
```

We are still actively polishing the package. You could look at this [tutorial](https://github.com/dmlc/minpy/blob/master/examples/demo/minpy_tutorial.ipynb) to understand its concept. Documents are hosted [here](https://minpy.readthedocs.io/en/latest/).
