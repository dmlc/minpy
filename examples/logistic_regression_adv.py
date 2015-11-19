import minpy
import minpy.numpy as np
import minpy.caffe as caffe

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1)

def predict(layer, inputs):
    # original code:
    # return sigmoid(np.dot(inputs, weights)
    return sigmoid(layer.ff(inputs))

def loss_func(layer, inputs, targets):
    def loss_theta(layer):
        pred = predict(layer, inputs)
        return -np.sum(np.log(pred * targets)) # negative log likelihood
    return loss_theta

num_samples = 10000
num_inputs = 128
num_outputs = 256
inputs = np.random.randn((num_samples, num_inputs))
targets = np.random.randn((num_samples, num_outputs))
# original code:
# weights = np.random.randn((num_inputs, num_outputs))
inner_product_layer = caffe.InnerProductLayer(
        input_shapes = (num_samples, num_inputs),
        num_outputs  = num_outputs)

grad_func = minpy.grad(loss_func(inner_product_layer, inputs, targets))

for i in range(100):
    inner_product_layer.get_learnable_params()[0] -= grad_func(inner_product_layer) * 0.01
