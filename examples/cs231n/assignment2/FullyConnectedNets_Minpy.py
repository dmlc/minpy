"""
  Pure python version of FullyConnectedNets_Minpy.ipynb
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net_minpy import TwoLayerNet, FullyConnectedNet # import minpy's model
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

def RunTwoLayerNet():
  model = TwoLayerNet()
  solver = Solver(model, data, optim_config={'learning_rate': 1e-3,}, lr_decay=0.95, print_every = 100)
  solver.train()

def RunFullyConnectedNet():
  N, D, H1, H2, C = 2, 15, 20, 30, 10
  X = np.random.randn(N, D)
  y = np.random.randint(C, size=(N,))
  
  for reg in [0, 3.14]:
    print 'Running check with reg = ', reg
    model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                      reg=reg, weight_scale=5e-2)
  
    loss, grads = model.loss_and_derivative(X, y)
    print 'Initial loss: ', loss
    ''' 
    for name in sorted(grads):
      f = lambda _: model.loss_and_derivative(X, y)[0]
      grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
      print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
    ''' 

#RunTwoLayerNet()
RunFullyConnectedNet()
