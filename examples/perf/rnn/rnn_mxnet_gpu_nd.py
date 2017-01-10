"""Benchmark vanilla RNN using MinPy CPU."""
import argparse
import time

import mxnet as mx
from minpy.nn.model import ModelBase
from six.moves import xrange

num_cold = 5

def rnn_step_forward(x, prev_h, Wx, Wh, b):
  next_h, cache = None, None
  next_h = mx.nd.tanh(mx.nd.dot(prev_h, Wh) + mx.nd.dot(x, Wx) + b)
  cache = next_h, prev_h, x, Wx, Wh
  return next_h, cache

def rnn_step_backward(dnext_h, cache):
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  # Load values from rnn_step_forward
  next_h, prev_h, x, Wx, Wh = cache
  # Gradients of loss wrt tanh
  dtanh = dnext_h * (1 - next_h * next_h)  # (N, H)
  # Gradients of loss wrt x
  dx = mx.nd.dot(dtanh, Wx, transpose_b=True) #dx = mx.nd.dot(dtanh, Wx.T)
  # Gradients of loss wrt prev_h
  dprev_h = mx.nd.dot(dtanh, Wh, transpose_b=True) #dprev_h = mx.nd.dot(dtanh, Wh.T)
  # Gradients of loss wrt Wx
  dWx = mx.nd.dot(x, dtanh, transpose_a=True)  #dWx = mx.nd.dot(x.T, dtanh)  # (D, H)
  # Gradients of loss wrt Wh
  dWh = mx.nd.dot(prev_h, dtanh, transpose_a=True) #dWh = mx.nd.dot(prev_h.T, dtanh)
  # Gradients of loss wrt b. Note we broadcast b in practice. Thus result of
  # matrix ops are just sum over columns
  db = mx.nd.sum(dtanh, axis=0)  # == np.ones([N, 1]).T.dot(dtanh)[0, :]
  return dx, dprev_h, dWx, dWh, db

def affine_forward(x, w, b):
  out = mx.nd.dot(x, w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  x, w, b = cache
  dx = mx.nd.dot(dout, w, transpose_b=True) #dx = mx.nd.dot(dout, w.T).reshape(x.shape)
  dw = mx.nd.dot(x, dout, transpose_a=True) #dw = mx.nd.dot(x.T, dout)
  db = mx.nd.sum(dout, axis=0)
  return dx, dw, db

def l2_loss(x, label):
    N = x.shape[0]
    C = x.shape[1]
    if len(label.shape) == 1:
        onehot_label = mx.nd.zeros([N, C], ctx=mx.gpu(0))
        mx.nd.onehot_encode(label, onehot_label)
    else:
        onehot_label = label
    loss = mx.nd.sum((x - onehot_label)**2) / N
    grad = 2 * (x - onehot_label) / N
    return loss, grad


class RNNNet(ModelBase):
    def __init__(self, args):
        super(RNNNet, self).__init__()
        self.add_param(name='Wx', shape=(args.input_size, args.hidden_size)) \
            .add_param(name='Wh', shape=(args.hidden_size, args.hidden_size))\
            .add_param(name='b', shape=(args.hidden_size,))                  \
            .add_param(name='Wa', shape=(args.hidden_size, args.num_classes))\
            .add_param(name='ba', shape=(args.num_classes,))
        self.num_unroll_steps = args.num_unroll_steps
        self.hshape = (args.batch_size, args.hidden_size)

    def forward(self, X, y):
        h = mx.nd.zeros(self.hshape, ctx=mx.gpu(0))  # init hidden state
        rnn_cache = []
        for t in xrange(self.num_unroll_steps):
            h, _ = rnn_step_forward(X, h, self.params['Wx'],
                                    self.params['Wh'], self.params['b'])
        predict, _ = affine_forward(h, self.params['Wa'], self.params['ba'])
        loss, _ = l2_loss(predict, y)
        return loss


    def ffbp(self, X, y):
        h = mx.nd.zeros(self.hshape, ctx=mx.gpu(0))  # init hidden state
        rnn_cache = []
        for t in xrange(self.num_unroll_steps):
            h, rnn_cache_t = rnn_step_forward(X, h, self.params['Wx'],
                                self.params['Wh'], self.params['b'])
            rnn_cache.append(rnn_cache_t)
        predict, affine_cache = affine_forward(h, self.params['Wa'], self.params['ba'])
        loss, grad = l2_loss(predict, y)

        daffine, dWa, dba = affine_backward(grad, affine_cache)

        dx = mx.nd.zeros((X.shape[0], X.shape[1]), ctx=mx.gpu(0))
        dWx = mx.nd.zeros((X.shape[1], daffine.shape[1]), ctx=mx.gpu(0))
        dWh = mx.nd.zeros((daffine.shape[1], daffine.shape[1]), ctx=mx.gpu(0))
        db = mx.nd.zeros((daffine.shape[1],), ctx=mx.gpu(0))

        dnext_h_t = daffine
        for t in xrange(self.num_unroll_steps):
            dx_t, dprev_h_t, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h_t, rnn_cache[t])
            dnext_h_t = dprev_h_t
        
            dx += dx_t
            dWx += dWx_t
            dWh += dWh_t
            db += db_t
        
        dx.wait_to_read()
        dWx.wait_to_read()
        dWh.wait_to_read()
        db.wait_to_read()


def main(args):
    # Create model.
    model = RNNNet(args)
    for k, v in model.param_configs.items():
        model.params[k] = mx.nd.zeros(v['shape'], ctx=mx.gpu(0))

    data = mx.nd.zeros((args.batch_size, args.input_size), ctx=mx.gpu(0)) # Data of only one time step.
    label = mx.nd.zeros((args.batch_size,), ctx=mx.gpu(0))

    for l in range(args.num_loops):
        if l == num_cold:
            start = time.time()
        if args.only_forward:
            l = model.forward(data, label) 
            l.wait_to_read()
        else:
            model.ffbp(data, label)
    dur = time.time() - start
    print('Per Loop Time: %.6f' % (dur / (args.num_loops - num_cold)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-forward', default=False, action='store_true')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--hidden-size', default=512, type=int)
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--num-classes', default=10, type=int)
    parser.add_argument('--num-unroll-steps', default=30, type=int)
    parser.add_argument('--num-loops', default=20, type=int)
    #import profile
    #profile.run('main(parser.parse_args())')
    main(parser.parse_args())
