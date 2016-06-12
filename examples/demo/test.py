import logging
import minpy.numpy as np
import minpy.dispatch.policy as ply
from minpy.core import grad_and_loss, function
import util
import mxnet as mx

logging.getLogger('minpy.array').setLevel(logging.WARN)

x = mx.symbol.Variable('x')
sm = mx.symbol.SoftmaxOutput(data=x, name='softmax', grad_scale=1/10000.0)

softmax = function(sm, [('x', (10000, 5)), ('softmax_label', (10000,))])

x, t = util.get_data()
#w = np.random.randn(500, 5)
w = util.get_weight()
softmax_label = np.argmax(t, axis=1)

def predict(w, x):
    '''
    a = np.exp(np.dot(x, w))
    a_sum = np.sum(a, axis=1, keepdims=True)
    prob = a / a_sum
    '''
    y = np.dot(x, w)
    prob = softmax(x=y, softmax_label=softmax_label)
    return prob

#util.plot_data(x, t)
#util.plot_data(x, predict(w, x))

'''
for i in range(1):
    prob = predict(w, x)
    #print prob
    dy = t - prob
    dw = np.dot(x.T, dy) / 10000
    w -= 0.1 * dw

print w

#util.plot_data(x, predict(w, x))

'''
def loss(w, x):
    prob = predict(w, x)
    return -np.sum(np.log(prob) * t) / 10000  + 0.5 * w * w

gl = grad_and_loss(loss)

for i in range(10):
    dw, loss = gl(w, x)
    print loss
    w -= 0.1 * dw
