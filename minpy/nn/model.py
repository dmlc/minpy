""" Model base class codes. Adapted from cs231n lab codes. """
import abc
import functools
import minpy

class ParamsNameNotFoundError(ValueError):
    """ Error of not existed name during accessing model params """
    pass

class UnknownAccessModeError(ValueError):
    """ Error of unexpected mode during accessing model params """
    pass

class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.params = {}
        self.param_configs = {}

    @abc.abstractmethod
    def forward(self, X):
        """ do forward and output the loss """
        return

    @abc.abstractmethod
    def loss(self, predict, y):
        return
