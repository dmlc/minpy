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
  model = FullyConnectedNet([100, 50], dropout=0.5)
  solver = Solver(model, data, optim_config={'learning_rate': 1e-3,}, lr_decay=0.95, print_every = 100)
  solver.train()

#RunTwoLayerNet()
RunFullyConnectedNet()
