import sys
from minpy import core
import minpy.mxnet as mx
import minpy.mxnet.ndarray as nd
import numpy.random as random

def sigmoid(x):
    return 1.0 / (1 + nd.exp(-x))

def predict(weights, inputs):
    return sigmoid(nd.dot(inputs, weights))

def training_loss(weights, inputs, targets):
    preds = predict(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -nd.sum(nd.log(label_probabilities))

xshape = (256, 500)
wshape = (500, 250)
tshape = (256, 250)
inputs = nd.array(random.rand(*xshape) - 0.5, ctx=mx.gpu(0))
targets = nd.array(random.randint(0, 2, size=tshape), ctx=mx.gpu(0))
weights = nd.array(random.rand(*wshape) - 0.5, ctx=mx.gpu(0))

training_gradient_fun = core.grad(training_loss)

print('Initial loss: {}'.format(training_loss(weights, inputs, targets).asnumpy()))
for i in range(100):
    gr = training_gradient_fun(weights, inputs, targets)
    #print('Training gradient: {}'.format(gr))
    weights -= gr * 0.1
    if i % 10 == 0:
        print('Trained loss: {}'.format(training_loss(weights, inputs, targets).asnumpy()))
