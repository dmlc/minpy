import abc
import minpy 
from minpy.array_variants import ArrayType

def NumpyVarToMinpy(var):
  return minpy.array.Value.wrap(var)

def MinpyVarToNumpy(var):
  return minpy.array.Value.wrap(var).get_data(ArrayType.NUMPY)

class ModelBase(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def loss_and_derivative(self, X, y):
    """ do forward and output the loss and derivative, if y is not none"""
    return

  def loss(self, *args, **kwargs):
    minpy_args = [NumpyVarToMinpy(v) for v in args]
    minpy_kwargs = {}
    for key, value in minpy_kwargs.iteritems():
      minpy_kwargs[key] = NumpyVarToMinpy(value)

    minpy_res = self.loss_and_derivative(*minpy_args, **minpy_kwargs)
    if len(minpy_res) == 1:
      minpy_res = [minpy_res]

    # TODO(Haoran): handle grad dicitonary
    numpy_res = [NumpyVarToMinpy(v) for v in minpy_res]
    if len(minpy_res) == 1:
      return numpy_res[0]
    else:
      return numpy_res[0], numpy_res[1]
    
