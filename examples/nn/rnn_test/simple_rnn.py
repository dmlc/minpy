import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

RNNState = namedtuple("RNNState", ["h"])
RNNParam = namedtuple("RNNParam", ["i2h_weight", "i2h_bias",
                                   "h2h_weight", "h2h_bias"])
RNNModel = namedtuple("RNNModel", ["rnn_exec", "symbol",
                                   "init_states", "last_states",
                                   "seq_data", "seq_labels", "seq_outputs",
                                   "param_blocks"])

def rnn(num_hidden, in_data, prev_state, param, seqidx, layeridx):
    i2h = mx.sym.FullyConnected(data=in_data,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    if seqidx > 0:
      h2h = mx.sym.FullyConnected(data=prev_state,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
      hidden = i2h + h2h
    else:
      hidden = i2h

    hidden = mx.sym.Activation(data=hidden, act_type="tanh")
    return RNNState(h=hidden)

def rnn_unroll(num_rnn_layer, seq_len, input_size,
                num_hidden, num_label):

    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    for i in range(num_rnn_layer):
        param_cells.append(RNNParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                    i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                    h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                    h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
    loss_all = []
    ori_data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    data_timestamp = mx.sym.SliceChannel(data=ori_data, num_outputs=seq_len, squeeze_axis=1)
    
    hidden = None
    for seqidx in range(seq_len):
        in_data = data_timestamp[seqidx]
        next_state = rnn(num_hidden, in_data=in_data,
            prev_state=hidden,
            param=param_cells[i],
            seqidx=seqidx, layeridx=i)
        hidden = next_state.h
    fc = mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias, num_hidden=num_label)
    reg = mx.sym.LinearRegressionOutput(data=fc, label=label)
    return reg
