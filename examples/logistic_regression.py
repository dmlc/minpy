import minpy
import minpy.numpy as np

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1)

def predict(weight, inputs):
    return sigmoid(np.dot(inputs, weight))

def loss_func(weight, inputs, targets):
    def loss_theta(weight):
        pred = predict(weight, inputs)
        return -np.sum(np.log(pred * targets))
    return loss_theta

num_samples = 10000
num_inputs = 128
num_outputs = 256
inputs = np.random.randn((num_samples, num_inputs))
targets = np.random.randn((num_samples, num_outputs))
weights = np.random.randn((num_inputs, num_outputs))

grad_func = minpy.grad(loss_func(weights, inputs, targets))

for i in range(100):
    weights -= grad_func(weights) * 0.01
