from minpy.core import grad
import minpy.numpy as np
import minpy.numpy.random as random
import minpy.dispatch.policy as policy
from minpy.context import Context, cpu, gpu, set_context

#np.set_policy(policy.OnlyNumpyPolicy())
set_context(gpu(1)) # set the global context as gpu(1)

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

with gpu(0):
    xshape = (256, 500)
    wshape = (500, 250)
    tshape = (256, 250)
    inputs = random.rand(*xshape) - 0.5
    targets = np.zeros(tshape)
    truth = random.randint(0, 250, 256)
    targets[np.arange(256), truth] = 1
    weights = random.rand(*wshape) - 0.5

    training_gradient_fun = grad(training_loss)

    for i in range(20):
        print('Trained loss accuracy #{}: {}%'.format(i, training_accuracy(weights, inputs)))
        gr = training_gradient_fun(weights, inputs)
        weights -= gr * 0.01
    print("\nff and bp on {0}".format(weights.context))

print("\nexecute on cpu")
with cpu():
    x_cpu = random.rand(32, 64) - 0.5
    y_cpu = random.rand(64, 32) - 0.5
    z_cpu = np.dot(x_cpu, y_cpu)
    print('z_cpu.context = {0}'.format(z_cpu.context))

print("\nexecute on gpu(0)")
with gpu(0):
    x_gpu0 = random.rand(32, 64) - 0.5
    y_gpu0 = random.rand(64, 32) - 0.5
    z_gpu0 = np.dot(x_gpu0, y_gpu0)
    z_gpu0.asnumpy()
    print('z_gpu0.context = {0}'.format(z_gpu0.context))

print("\n[use global context] execute on gpu(1)")
x_gpu1 = random.rand(32, 64) - 0.5
y_gpu1 = random.rand(64, 32) - 0.5
z_gpu1 = np.dot(x_gpu1, y_gpu1)
z_gpu1.asnumpy()
print('z_gpu1.context = {0}'.format(z_gpu1.context))
