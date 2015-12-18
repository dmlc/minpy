from minpy import core
import minpy.numpy as np
import numpy

def sigmoid(x):
    return np.multiply(0.5, np.add(np.tanh(x), 1))

def predict(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights, inputs):
    preds = predict(weights, inputs)
    label_probabilities = np.subtract(np.subtract(np.add(1, np.multiply(2, np.multiply(preds, targets))), preds), targets)
    return np.negative(np.sum(np.log(label_probabilities)))

inputs = numpy.asarray([
    [0.52, 1.12,  0.77],
    [0.88, -1.08, 0.15],
    [0.52, 0.06, -1.30],
    [0.74, -2.49, 1.39]])
targets = numpy.asarray([True, True, False, True])

training_gradient_fun = core.grad(training_loss)

weights = numpy.asarray([0.0, 0.0, 0.0])

print('Initial loss: {}'.format(training_loss(weights, inputs)))
for i in range(100):
    gr = training_gradient_fun(weights, inputs)
    print('Training gradient: {}'.format(gr))
    weights -= gr * 0.01
print('Trained loss: {}'.format(training_loss(weights, inputs)))
