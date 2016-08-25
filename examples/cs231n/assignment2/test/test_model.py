"""
  Pure python version of FullyConnectedNets_Minpy.ipynb
"""
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net_minpy import TwoLayerNet, FullyConnectedNet  # import minpy's model
from cs231n.classifiers.cnn_minpy import ThreeLayerConvNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
from cs231n.layers import svm_loss_forward


def RunTwoLayerNet():
    model = TwoLayerNet()
    solver = Solver(model,
                    data,
                    optim_config={'learning_rate': 1e-4,},
                    lr_decay=0.95,
                    print_every=100)
    solver.train()


def RunFullyConnectedNet():
    model = FullyConnectedNet([100, 50], dropout=0.5, use_batchnorm=True)
    solver = Solver(model,
                    data,
                    optim_config={'learning_rate': 4e-3,},
                    lr_decay=0.95,
                    print_every=100)
    solver.train()


def RunCnnNet():
    model = ThreeLayerConvNet(weight_scale=0.001, reg=0.001)

    solver = Solver(model,
                    data,
                    num_epochs=10,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-4,
                        'momemtum': 0.9,
                    },
                    verbose=True,
                    print_every=20)
    solver.train()


def Debug():
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print 'Running check with reg = ', reg
        model = FullyConnectedNet([H1, H2],
                                  input_dim=D,
                                  num_classes=C,
                                  reg=reg,
                                  weight_scale=5e-2,
                                  dtype=np.float64,
                                  conv_mode='numpy')

        loss, grads = model.loss(X, y)
        print 'Initial loss: ', loss
        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f,
                                               model.params[name].asnumpy(),
                                               verbose=False,
                                               h=1e-5)
            print '%s relative error: %.2e' % (name, rel_error(grad_num,
                                                               grads[name]))



def SMALL_CNN():
    num_train = 100
    small_data = {
      'X_train': data['X_train'][:num_train],
      'y_train': data['y_train'][:num_train],
      'X_val': data['X_val'],
      'y_val': data['y_val'],
    }
    
    model = ThreeLayerConvNet(weight_scale=1e-3)
    
    solver = Solver(model, small_data,
                    num_epochs=10,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    verbose=True,
                    print_every=20)

    solver.train()

data = get_CIFAR10_data()
for k, v in data.iteritems():
    print '%s: ' % k, v.shape

RunTwoLayerNet()
RunFullyConnectedNet()
RunCnnNet()
SMALL_CNN()
