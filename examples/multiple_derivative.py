import minpy 
import minpy.array
from minpy.array_variants import ArrayType

from minpy.core import grad
import minpy.numpy as np
import minpy.numpy.random as random
import minpy.dispatch.policy as policy

#np.set_policy(policy.OnlyNumpyPolicy())

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
    error = np.count_nonzero(np.argmax(preds, axis=1) - np.argmax(targets, axis=1))
    return (256 - error) * 100 / 256.0

xshape = (256, 500)
wshape = (500, 250)
tshape = (256, 250)
inputs = random.rand(*xshape) - 0.5
targets = np.zeros(tshape)
truth = random.randint(0, 250, 256)
targets[np.arange(256), truth] = 1
weights = random.rand(*wshape) - 0.5

training_gradient_fun_0 = grad(training_loss, 0)
print 'derivative of 1st argument by calling single arg-grad func'
print training_gradient_fun_0(weights, inputs)

training_gradient_fun_1 = grad(training_loss, 1)
print 'derivative of 2nd argument by calling single arg-grad func'
print training_gradient_fun_1(weights, inputs)

training_gradient_fun_both = grad(training_loss, [0, 1])
v = training_gradient_fun_both(weights, inputs)
print 'derivative of 1st argument by calling multiple arg-grad func'
print v[0]
print 'derivative of 2nd argument by calling multiple arg-grad func'
print v[1]
