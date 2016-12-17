from minpy import core
import minpy.numpy as np
import minpy.numpy.random as random
import mxnet as mx


def test_mxnet_logistic():
    def sigmoid(x):
        return np.multiply(0.5, np.add(np.tanh(x), 1))

    xshape = (256, 500)
    #needs to reverse. because of mxnet's setting
    wshape = (250, 500)
    tshape = (256, 250)
    inputs = random.rand(*xshape) - 0.5
    targets = np.zeros(tshape)
    truth = random.randint(0, 250, 256)
    targets[np.arange(256), truth] = 1
    weights = np.random.rand(*wshape) - 0.5

    x = mx.sym.Variable(name='x')
    fc = mx.sym.FullyConnected(name='fc', data=x, num_hidden=250)
    act = mx.sym.Activation(data=fc, act_type='sigmoid')

    f = core.Function(act, {'x': xshape})

    def predict(weights, inputs):
        #return f( data=[('x', inputs)], weight=[('fc_weight', weights)], ctx=mx.cpu())
        return f(x=inputs, fc_weight=weights)

    def training_loss(weights, inputs):
        preds = predict(weights, inputs)
        label_probabilities = preds * targets + (1 - preds) * (1 - targets)
        return -np.sum(np.log(label_probabilities))

    training_gradient_fun = core.grad(training_loss)

    print('Initial loss: {}'.format(training_loss(weights, inputs)))
    for i in range(100):
        gr = training_gradient_fun(weights, inputs)
        #print('Training gradient: {}'.format(gr))
        weights -= gr * 0.1
        if i % 10 == 0:
            print('Trained loss: {}'.format(training_loss(weights, inputs)))

    # The training loss should be around 300 in a bug-free Minpy
    if (training_loss(weights, inputs)[0] > 600):
        assert (False)


if __name__ == "__main__":
    test_mxnet_logistic()
