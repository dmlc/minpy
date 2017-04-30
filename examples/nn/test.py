import mxnet as mx
from mxnet.symbol import *
from minpy.nn.layers import softmax_loss

context = mx.gpu(0)

data_shape = (128, 3, 32, 32)
def bind(symbol, context=context, shape=None):
    global data_shape
    if shape is None: shape = data_shape
    symbol = symbol(Variable('data', shape=shape))
    _, output_shapes, _ = symbol.infer_shape(data=shape)
    data_shape = output_shapes[0]

    executor = symbol.simple_bind(context)

    for key, value in executor.arg_dict.items():
        executor.arg_dict[key][:] = mx.nd.random_normal(0, 1, value.shape)
    for key, value in executor.aux_dict.items():
        executor.aux_dict[key][:] = mx.nd.random_normal(0, 1, value.shape)

    return executor

def convolution(**kwargs):
    default_kwargs = {'kernel' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)}
    default_kwargs.update(kwargs)
    return Convolution(**default_kwargs)

def convolution_module(**kwargs):
    return [
        bind(lambda data : convolution(data=data, **kwargs)),
        bind(lambda data : BatchNorm(data=data, fix_gamma=False)),
        bind(lambda data : Activation(data=data, act_type='relu')),
    ]

def forward(sequence, data):
    return reduce(lambda X, executor : executor.forward(data=X, is_train=True)[0], sequence, data)

def backward(sequence, gradient):
    for executor in reversed(sequence):
        executor.backward(out_grads=gradient)
        gradient = executor.grad_dict['data']

N = 12
network = convolution_module(num_filter=16)
for i in range(N):
    network.extend(convolution_module(num_filter=16))
network.extend(convolution_module(num_filter=32, stride=(2, 2)))
for i in range(N):
    network.extend(convolution_module(num_filter=32))
network.extend(convolution_module(num_filter=64, stride=(2, 2)))
for i in range(N):
    network.extend(convolution_module(num_filter=64))
network.extend([
    bind(lambda data : Pooling(data=data, pool_type='avg', kernel=(8, 8), stride=(1, 1), pad=(0, 0))),
    bind(lambda data : Flatten(data=data)),
    bind(lambda data : FullyConnected(data=data, num_hidden=10)),
])

data = Variable('data', shape=(128, 10))
labels = Variable('labels', shape=(128,))
loss_symbol = SoftmaxOutput(data=data, label=labels)
loss = loss_symbol.simple_bind(context)

import numpy as np

X = mx.nd.random_normal(0, 1, (50000, 3, 32, 32), context)
Y = mx.nd.array(np.random.choice(np.arange(10), (50000,)), context)
iterator = mx.io.NDArrayIter(data=X, label=Y, batch_size=128)

while True:
    iterator.reset()
    for i, batch in enumerate(iterator):
        data, labels = batch.data[0], batch.label[0]
        scores = forward(network, data)
        results = loss.forward(data=scores, labels=labels, is_train=True)
        softmax_loss(results[0].asnumpy(), labels.asnumpy())
        loss.backward()
        backward(network, loss.grad_dict['data'])

        if (i + 1) % 100 == 0:
            print i + 1
