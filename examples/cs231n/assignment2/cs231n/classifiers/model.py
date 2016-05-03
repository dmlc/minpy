import abc
import functools 
import minpy 
from minpy.array_variants import ArrayType

class ModelBase(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def loss_and_derivative(self, X, y):
    """ do forward and output the loss and derivative, if y is not none"""
    return
