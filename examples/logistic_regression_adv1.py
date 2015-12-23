from minpy import core
import minpy.mxnet as mx
import minpy.mxnet.nd as nd
import minpy.mxnet.rnd as rnd

def sigmoid(x):
    return 1.0 / (1 + nd.exp(-x))

def predict(weights, inputs):
    return sigmoid(nd.dot(inputs, weights))

def training_loss(weights, inputs):
    preds = predict(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -nd.sum(nd.log(label_probabilities))

xshape = (256, 500)
wshape = (500, 250)
tshape = (256, 250)
inputs = random.rand(*xshape, ctx=mx.gpu(0)) - 0.5
targets = random.randint(0, 2, size=tshape, ctx=mx.gpu(0))
weights = random.rand(*wshape, ctx=mx.gpu(0)) - 0.5

training_gradient_fun = core.grad(training_loss)

print('Initial loss: {}'.format(training_loss(weights, inputs)))
for i in range(100):
    gr = training_gradient_fun(weights, inputs)
    #print('Training gradient: {}'.format(gr))
    weights -= gr * 0.1
    if i % 10 == 0:
        print('Trained loss: {}'.format(training_loss(weights, inputs)))
