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

    def add_param(self, name, shape, **kwargs):
        assert(name not in self.param_configs), 'Duplicate parameter name %s' % name
        self.param_configs[name] = { 'shape': shape }
        self.param_configs[name].update(kwargs)
        return self

    def add_params(self, param_dict):
        for name, pconfig in param_dict.items():
            assert(name not in self.param_configs), 'Duplicate parameter name %s' % name
            self.param_configs[name] = pconfig
        return self

    @abc.abstractmethod
    def forward(self, X):
        """ do forward and output the loss """
        return

    @abc.abstractmethod
    def loss(self, predict, y):
        return
