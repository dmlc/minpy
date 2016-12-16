"""Benchmark vanilla RNN using MinPy CPU."""
import argparse
import time

import numpy as np
from minpy.nn.model import ModelBase
from six.moves import xrange

num_cold = 5

def rnn_step_forward(x, prev_h, Wx, Wh, b):
  next_h, cache = None, None
  next_h = np.tanh(prev_h.dot(Wh) + x.dot(Wx) + b)
  cache = next_h, prev_h, x, Wx, Wh
  return next_h, cache

def rnn_step_backward(dnext_h, cache):
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  # Load values from rnn_step_forward
  next_h, prev_h, x, Wx, Wh = cache
  # Gradients of loss wrt tanh
  dtanh = dnext_h * (1 - next_h * next_h)  # (N, H)
  # Gradients of loss wrt x
  dx = dtanh.dot(Wx.T)
  # Gradients of loss wrt prev_h
  dprev_h = dtanh.dot(Wh.T)
  # Gradients of loss wrt Wx
  dWx = x.T.dot(dtanh)  # (D, H)
  # Gradients of loss wrt Wh
  dWh = prev_h.T.dot(dtanh)
  # Gradients of loss wrt b. Note we broadcast b in practice. Thus result of
  # matrix ops are just sum over columns
  db = dtanh.sum(axis=0)  # == np.ones([N, 1]).T.dot(dtanh)[0, :]
  return dx, dprev_h, dWx, dWh, db

def affine_forward(x, w, b):
  out = x.reshape(x.shape[0], -1).dot(w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  x, w, b = cache
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x.reshape(x.shape[0], -1).T.dot(dout)
  db = np.sum(dout, axis=0)
  return dx, dw, db

def l2_loss(x, label):
    N = x.shape[0]
    C = x.shape[1]
    if len(label.shape) == 1:
        onehot_label = np.zeros([N, C])
        #np.onehot_encode(label, onehot_label)
    else:
        onehot_label = label
    loss = np.sum((x - onehot_label)**2) / N
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
        h = np.zeros(self.hshape)  # init hidden state
        rnn_cache = []
        for t in xrange(self.num_unroll_steps):
            h, _ = rnn_step_forward(X, h, self.params['Wx'],
                                self.params['Wh'], self.params['b'])
        predict, _ = affine_forward(h, self.params['Wa'], self.params['ba'])
        loss, _ = l2_loss(predict, y)
        return loss


    def ffbp(self, X, y):
        h = np.zeros(self.hshape)  # init hidden state
        rnn_cache = []
        for t in xrange(self.num_unroll_steps):
            h, rnn_cache_t = rnn_step_forward(X, h, self.params['Wx'],
                                self.params['Wh'], self.params['b'])
            rnn_cache.append(rnn_cache_t)
        predict, affine_cache = affine_forward(h, self.params['Wa'], self.params['ba'])
        loss, grad = l2_loss(predict, y)

        daffine, dWa, dba = affine_backward(grad, affine_cache)

        dx = np.zeros([X.shape[0], X.shape[1]])
        dWx = np.zeros([X.shape[1], daffine.shape[1]])
        dWh = np.zeros([daffine.shape[1], daffine.shape[1]])
        db = np.zeros(daffine.shape[1])

        dnext_h_t = daffine
        for t in xrange(self.num_unroll_steps):
            dx_t, dprev_h_t, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h_t, rnn_cache[t])
            dnext_h_t = dprev_h_t
        
            dx += dx_t
            dWx += dWx_t
            dWh += dWh_t
            db += db_t


def main(args):
    # Create model.
    model = RNNNet(args)
    for k, v in model.param_configs.items():
        model.params[k] = np.zeros(v['shape'])

    data = np.zeros((args.batch_size, args.input_size)) # Data of only one time step.
    label = np.zeros((args.batch_size,), dtype=np.int)

    for l in range(args.num_loops):
        if l == num_cold:
            start = time.time()
        if args.only_forward:
            model.forward(data, label) 
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
    main(parser.parse_args())
