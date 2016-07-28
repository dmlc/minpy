""" Simple multi-layer perception neural network using Minpy """
import sys
import minpy
import numpy as np
from minpy.utils.data_utils import get_CIFAR10_data
from minpy.nn.io import NDArrayIter

def main(_):
  data = get_CIFAR10_data()
  # reshape all data to matrix
  data['X_train'] = data['X_train'].reshape([data['X_train'].shape[0], 3 * 32 * 32])
  data['X_val'] = data['X_val'].reshape([data['X_val'].shape[0], 3 * 32 * 32])
  data['X_test'] = data['X_test'].reshape([data['X_test'].shape[0], 3 * 32 * 32])
  
  train_data = data['X_train']
  dataiter = NDArrayIter(data['X_train'],
                         data['y_train'],
                         100,
                         True)

  for each_data in dataiter:
    print each_data



if __name__ == '__main__':
  main(sys.argv) 

