from minpy import core
import minpy.numpy as np
import minpy.numpy.random as random
import mxnet as mx


def sigmoid(x):
    return np.multiply(0.5, np.add(np.tanh(x), 1))

#xshape = (256, 500)
xshape = (256, 1, 30, 30)
inputs = random.rand(*xshape) - 0.5

data = mx.symbol.Variable(name='x')
conv1 = mx.symbol.Convolution(
    name='conv', data=data, kernel=(5, 5), pad=(1, 1), num_filter=5)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(
    data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))

flatten = mx.symbol.Flatten(data=pool1)
fc = mx.sym.FullyConnected(name='fc', data=flatten, num_hidden=250)
act = mx.sym.Activation(data=fc, act_type='sigmoid')

f = core.Function(act, {'x': xshape})


targets = np.zeros(f.get_one_output_shape())
truth = random.randint(0, 250, 256)
targets[np.arange(256), truth] = 1

param_shapes = f._param_shapes
fc_weight = random.rand(*param_shapes['fc_weight']) - 0.5
fc_bias = np.zeros(param_shapes['fc_bias']) * 0
conv_weight = random.rand(*param_shapes['conv_weight']) - 0.5
conv_bias = np.zeros(param_shapes['conv_bias']) * 0

def predict(inputs, fc_weight, fc_bias, conv_weight, conv_bias):
    #return f( data=[('x', inputs)], weight=[('fc_weight', weights)], ctx=mx.cpu())
    return f(x=inputs,
             fc_weight=fc_weight,
             fc_bias=fc_bias,
             conv_weight=conv_weight,
             conv_bias=conv_bias)


def training_loss(inputs, targets, fc_weight, fc_bias, conv_weight, conv_bias):
    preds = predict(inputs, fc_weight, fc_bias, conv_weight, conv_bias)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))


training_gradient_fun = core.grad_and_loss(training_loss, range(2, 6))

lr = 1e-5
for i in range(100):
    grads, loss = training_gradient_fun(inputs, targets, fc_weight, fc_bias,
                                        conv_weight, conv_bias)
    #print('Training gradient: {}'.format(gr))
    fc_weight -= grads[0] * lr
    fc_bias -= grads[1] * lr
    conv_weight -= grads[2] * lr
    conv_bias -= grads[3] * lr
    if i % 10 == 0:
        print('Trained loss: {}'.format(loss))
