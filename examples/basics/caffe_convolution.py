from minpy import core
import minpy.numpy as np
import minpy.numpy.random as random
import mxnet as mx


def sigmoid(x):
    return np.multiply(0.5, np.add(np.tanh(x), 1))

#xshape = (256, 500)
xshape = (256, 1, 30, 30)
#needs to reverse. because of mixnet's setting
tshape = (256, 250)
inputs = random.rand(*xshape) - 0.5
targets = np.zeros(tshape)
truth = random.randint(0, 250, 256)
targets[np.arange(256), truth] = 1

fc_wshape = (250, 845)
fc_weight = random.rand(*fc_wshape) - 0.5

fc_bshape = (250, )
#fc_bias = np.zeros(*fc_bshape)
fc_bias = random.rand(*fc_bshape) * 0

conv_wshape = (5, 1, 5, 5)
conv_weight = random.rand(*conv_wshape) - 0.5

conv_bshape = (5, )
#conv_bias = np.zeros(*conv_bshape)
conv_bias = random.rand(*conv_bshape) * 0

data = mx.symbol.Variable(name='x')
conv1 = mx.symbol.CaffeOp(
    name='conv',
    num_weight=2,
    data_0=data,
    prototxt="layer {type:\"Convolution\" convolution_param { num_output: 5 kernel_size: 5 stride: 1} }")
act1 = mx.symbol.CaffeOp(data_0=conv1, prototxt="layer {type:\"TanH\"}")
pool1 = mx.symbol.CaffeOp(
    data_0=act1,
    prototxt="layer {type:\"Pooling\" pooling_param { pool: MAX kernel_size: 2 stride: 2}}")

fc1 = mx.symbol.CaffeOp(
    name='fc',
    num_weight=2,
    data_0=pool1,
    prototxt="layer {type:\"InnerProduct\" inner_product_param{num_output: 250} }")
act2 = mx.symbol.CaffeOp(data_0=fc1, prototxt="layer {type:\"Sigmoid\"}")

f = core.Function(act2, {'x': xshape})


def predict(inputs, fc_weight, fc_bias, conv_weight, conv_bias):
    #return f( data=[('x', inputs)], weight=[('fc_weight', weights)], ctx=mx.cpu())
    return f(x=inputs,
             fc_0_weight=fc_weight,
             fc_1_bias=fc_bias,
             conv_0_weight=conv_weight,
             conv_1_bias=conv_bias)


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
