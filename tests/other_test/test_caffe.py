import numpy as np
import mxnet as mx
from minpy import core
import minpy.nn.layers as layers
import minpy.utils.gradient_checker as gradient_checker
import minpy.dispatch.policy as plc

def test_caffe_concat():
    xshape_0 = (10, 40)
    xshape_1 = (10, 30)
    fake_y = np.zeros([10, 70])
    fake_y[:,0] = 1
    x_1 = rng.randn(*xshape_1) - 0.5

    inputs_0 = mx.sym.Variable(name='x_0')
    inputs_1 = mx.sym.Variable(name='x_1')
    concat = mx.symbol.CaffeOp(data_0=inputs_0,
                               data_1=inputs_1,
                               num_data=2,
                               prototxt="layer {type:\"Concat\"}")

    f = core.function(concat, {'x_0': xshape_0, 'x_1': xshape_1})

    def check_fn(x_0):
        return layers.l2_loss(f(x_0=x_0, x_1=x_1), fake_y)

    x_0 = rng.randn(*xshape_0) - 0.5
    gradient_checker.quick_grad_check(check_fn, x_0, rs=rng)

if __name__ == "__main__":
    test_caffe_concat()
