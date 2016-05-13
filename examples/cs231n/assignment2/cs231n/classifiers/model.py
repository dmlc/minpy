import abc
import functools 
import minpy 
from minpy.array_variants import ArrayType
from minpy.core import converter, MinpyVarToNumpy


class ModelBase(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, conv_mode = 'lazy'):
    self.convert_mode = conv_mode

  def loss(self, X, y = None):
    res = converter(self.convert_mode)(self.loss_and_derivative)(X, y)

    # make loss or score, i.e. res[0], return as numpy.float
    # while grads, i.e. res[1], could be minpy's array
    if (self.convert_mode == 'lazy'):
      if type(res) is not tuple:
        res = MinpyVarToNumpy(res)
      else:
        return MinpyVarToNumpy(res[0]), res[1]
    return res

  @abc.abstractmethod
  def loss_and_derivative(self, X, y):
    """ do forward and output the loss and derivative, if y is not none"""
    return
