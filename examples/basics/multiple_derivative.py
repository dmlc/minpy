from __future__ import absolute_import
from __future__ import print_function
import minpy
import minpy.array
from minpy.array_variants import ArrayType

from minpy.core import grad
from minpy.core import grad_and_loss
import minpy.numpy as np
import minpy.numpy.random as random

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2) + 1)


def predict(weights, inputs):
    return sigmoid(np.dot(inputs, weights))


def training_loss(weights, inputs):
    preds = predict(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    l = -np.sum(np.log(label_probabilities))
    return l


def training_accuracy(weights, inputs):
    preds = predict(weights, inputs)
    error = np.count_nonzero(
        np.argmax(
            preds, axis=1) - np.argmax(
                targets, axis=1))
    return (256 - error) * 100 / 256.0


xshape = (256, 500)
wshape = (500, 250)
tshape = (256, 250)
inputs = random.rand(*xshape) - 0.5
targets = np.zeros(tshape)
truth = random.randint(0, 250, 256)
targets[np.arange(256), truth] = 1
weights = random.rand(*wshape) - 0.5

#training_gradient_fun_0 = grad(training_loss, 0)
grad_arg0 = grad_and_loss(training_loss, 0)
grad, loss = grad_arg0(weights, inputs)
print('1st arg\'s grad by single grad func', grad)

grad_arg1 = grad_and_loss(training_loss, 1)
grad, loss = grad_arg1(weights, inputs)
print('2nd arg\'s grad by single grad func', grad)

grad_args = grad_and_loss(training_loss, [0, 1])
grads, loss = grad_args(weights, inputs)
print('1st arg\'s grad by single grad func', grads[0])
print('2nd arg\'s grad by single grad func', grads[1])
