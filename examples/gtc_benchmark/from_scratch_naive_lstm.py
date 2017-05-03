import mxnet.ndarray as nd

def linear(X, W, bias):
    return nd.dot(X, W) + bias

def sigmoid(x):
    return .5 * (nd.tanh(.5 * x) + 1)

import mxnet as mx
context = mx.gpu(1)

from mxnet.context import Context
Context.default_ctx = context

def gaussian(shape):
    return nd.random_normal(shape=shape).as_in_context(context)

D = 1024

WX_SHAPE = (7, D)
Wxi = gaussian(shape=WX_SHAPE)
Wxf = gaussian(shape=WX_SHAPE)
Wxo = gaussian(shape=WX_SHAPE)
Wxg = gaussian(shape=WX_SHAPE)

bX_SHAPE = (D,)
bxi = nd.zeros(shape=bX_SHAPE)
bxf = nd.zeros(shape=bX_SHAPE)
bxo = nd.zeros(shape=bX_SHAPE)
bxg = nd.zeros(shape=bX_SHAPE)

WH_SHAPE = (D, D)
Whi = gaussian(shape=WH_SHAPE)
Whf = gaussian(shape=WH_SHAPE)
Who = gaussian(shape=WH_SHAPE)
Whg = gaussian(shape=WH_SHAPE)

bH_SHAPE = (D,)
bhi = nd.zeros(shape=bH_SHAPE)
bhf = nd.zeros(shape=bH_SHAPE)
bho = nd.zeros(shape=bH_SHAPE)
bhg = nd.zeros(shape=bH_SHAPE)

W = gaussian(shape=(D, 10))
b = nd.zeros(shape=(10,))

N = 128
X = gaussian(shape=(N, 784 / 7, 7))

from time import time

t0 = time()
for index in range(10):
    h = nd.zeros((N, D))
    c = nd.zeros((N, D))

    for i in range(784 / 7):
        patch = mx.nd.slice_axis(X, axis=1, begin=i, end=(i + 1))
        i = sigmoid(linear(patch, Wxi, bxi) + linear(h, Whi, bhi))
        f = sigmoid(linear(patch, Wxf, bxf) + linear(h, Whf, bhf))
        o = sigmoid(linear(patch, Wxo, bxo) + linear(h, Who, bho))
        g = nd.tanh(linear(patch, Wxg, bxg) + linear(h, Whg, bhg))
        c = f * c + i * g
        h = o * mx.nd.tanh(c)
    linear(h, W, b).asnumpy()
print (time() - t0) / (index + 1)
