from minpy.core import grad
import minpy.numpy as np
import minpy.numpy.random as random
import minpy.dispatch.policy as policy

#np.set_policy(policy.OnlyNumpyPolicy())

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2) + 1)

def predict(weights, bias, inputs):
    return sigmoid(np.dot(inputs, weights) + bias)

def training_loss(weights, bias, inputs):
    preds = predict(weights, bias, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    l = -np.sum(np.log(label_probabilities))
    return l

def training_accuracy(weights, bias, inputs):
    preds = predict(weights, bias, inputs)
    error = np.count_nonzero(np.argmax(preds, axis=1) - np.argmax(targets, axis=1))
    return (256 - error) * 100 / 256.0

xshape = (256, 500)
wshape = (500, 250)
bshape = (250)
tshape = (256, 250)
inputs = random.rand(*xshape) - 0.5
targets = np.zeros(tshape)
truth = random.randint(0, 250, 256)
targets[np.arange(256), truth] = 1
weights = random.rand(*wshape) - 0.5
#bias = random.rand(bshape) - 0.5
#print bias.shape
bias = np.zeros(bshape)
print bias.shape

training_gradient_fun = grad(training_loss)

for i in range(20):
    print('Trained loss accuracy #{}: {}%'.format(i, training_accuracy(weights, bias, inputs)))
    gr = training_gradient_fun(weights, bias, inputs)
    weights -= gr * 0.01
