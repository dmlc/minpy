from minpy.core import grad
import minpy.numpy as np
import minpy.numpy.random as random

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1)

def predict(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights, inputs):
    preds = predict(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

xshape = (256, 500)
wshape = (500, 250)
tshape = (256, 250)
inputs = random.rand(*xshape) - 0.5
targets = random.randint(0, 2, size=tshape)
weights = random.rand(*wshape) - 0.5

training_gradient_fun = grad(training_loss)

print('Initial loss: {}'.format(training_loss(weights, inputs)))
for i in range(100):
    gr = training_gradient_fun(weights, inputs)
    #print('Training gradient: {}'.format(gr))
    weights -= gr * 0.1
    if i % 10 == 0:
        print('Trained loss: {}'.format(training_loss(weights, inputs)))
