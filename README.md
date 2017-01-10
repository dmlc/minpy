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
![p1](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/p1.png)
In symbolic programming, the control dependency before the print statement is required, otherwise the print operator will not appear on the critical dependency path and thus not being executed. As a contrast, MinPy's way is very straight-forward, just like what you've done in Python's hello world.

### Data-dependent branches
![p2](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/p2.png)
In symbolic programming, the `lambda` is required in each branch to lazily expand the dataflow graph during runtime, which is quite confusing. As a contrast, you could feel free to use the if statement in MinPy.

Tensorflow is just one typical example, many other packages (e.g. Theano, or even MXNet) have similar problems. The underlying reason is the trade-off between *symbolic programming* and *imperative programming*. Codes in symbolic programs (Tensorflow and Theano) are *not* doing concrete computation but generating dataflow graph instead. This enables extensive optimizations but requires reinventing almost all language constructs (like if and loop). Imperative programs (NumPy) generates dataflow graph *along with* the computation, so you could freely query or use the value just computed. For a rough analogy, symbolic programming is like C++, where you have two stages: compilation (generating dataflow graph) and execution (concrete evaluation), while imperative programming is like Python -- it is easy to tweak but less efficient.

In MinPy, we use NumPy syntax to ease your programming, while achieving good performance at the same time.

## Dynamic automatic gradient computation
Automatic gradient computation has become an essential component in modern deep learning systems. In MinPy, we adopts [Autograd](https://github.com/HIPS/autograd)'s approach to compute gradients. Since the dataflow graph is generated along with the computation, all kinds of native control flow are supported during gradient computation. For example:
```python
import minpy
from minpy.core import grad

def foo(x):
  if x >= 0:
    return x
  else:
    return 2 * x

foo_grad = grad(foo)
print foo_grad(3)  # should print 1.0
print foo_grad(-1) # should print 2.0
```
Here, feel free to use native `if` statement. A complete tutorial about auto-gradient computation could be found [here](https://minpy.readthedocs.io/en/latest/tutorial/autograd_tutorial.html).

## Elegant fallback for missing operators
You never like `NotImplementedError`, so do we. In MinPy, we automatically fallback to NumPy if some operators have not been implemented in MXNet yet. For example, the following code runs smoothly and you don't need to worry about copying arrays back and forth from GPU to CPU.
```python
import minpy.numpy as np
x = np.zeros((2, 3))
y = np.ones((2, 3))
z = np.logaddexp(x, y)
```

## Seamless MXNet symbol support
Although we pick the imperative side, we understand that symbolic programming is necessary for operators like convolution. Therefore, MinPy allows you to "wrap" a symbol into a function that could be called together with other imperative calls:
```python
import mxnet as mx
import minpy.numpy as np
from minpy.core import Function
# Create Function from symbol.
net = mx.sym.Variable('x')
net = mx.sym.Convolution(net, name='conv', kernel=(3, 3), num_filter=32, no_bias=True)
conv = Function(net, input_shapes={'x', (8, 3, 10, 10)}
# Call Function as normal function.
x = np.zeros((8, 3, 10, 10))
w = np.ones((32, 3, 3, 3,))
y = np.exp(conv(x=x, conv_weight=w))
```

## Is MinPy fast?
The imperative interface does raise many challenges, but MinPy is managed to achieve good speed when the workload is not too tiny.
![benchmark](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/benchmark.png)


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
