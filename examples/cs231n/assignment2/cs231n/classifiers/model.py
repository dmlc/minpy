import abc

class ModelBase(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def loss(self, X, y):
    """ do forward and output the loss and derivative, if y is not none"""
    return
