import minpy
import minpy.numpy as np
import minpy.caffe as caffe

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1)

def predict(weight, layer, inputs):
    # original code:
    # return sigmoid(np.dot(inputs, weight)
    return sigmoid(layer.ff(inputs, weight))

def loss_func(weight, layer, inputs, targets):
    def loss_theta(layer):
        pred = predict(weight, layer, inputs)
        return -np.sum(np.log(pred * targets)) # negative log likelihood
    return loss_theta

num_samples = 10000
num_inputs = 128
num_outputs = 256
inputs = np.random.randn((num_samples, num_inputs))
targets = np.random.randn((num_samples, num_outputs))
weights = np.random.randn((num_inputs, num_outputs))

# additional caffe layer statement
inner_product_layer = caffe.InnerProductLayer(
        input_shapes = (num_samples, num_inputs),
        num_outputs  = num_outputs)

# original code:
# grad_func = minpy.grad(loss_func(weights, inputs, targets))
grad_func = minpy.grad(loss_func(weights, inner_product_layer, inputs, targets))

for i in range(100):
    weights -= grad_func(weights) * 0.01
