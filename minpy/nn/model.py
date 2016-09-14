""" Model base class codes. Adapted from cs231n lab codes. """
import abc
import functools
import minpy
import numpy
import h5py

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
        self.aux_params = {}
        self.aux_param_configs = {}

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

    def add_aux_param(self, name, value):
        """ Add auxiliary parameter.

        Auxiliary parameter is the parameter not updated by back propagation. This function
        will add variable "name" into dictionary self.aux_params, which can be accessed by
        model in a solver.

        :param name: name of the parameter.
        :param value: value of the parameter.
        :return: model itself
        """
        assert(name not in self.aux_param_configs), 'Duplicate auxiliary parameter name %s' % name
        self.aux_param_configs[name] = value
        return self

    @abc.abstractmethod
    def forward(self, X, mode):
        """  do forward and output the loss

        :param X: input vector
        :param mode: a mode string either 'train' or 'test'
        :return: output vector
        """
        return

    @abc.abstractmethod
    def loss(self, predict, y):
        return

    def save(self, prefix):
        """Save model params into file.

        :param prefix: prefix of model name.
        """
        param_name = '%s.params' % prefix
        with h5py.File(param_name, 'w') as hf:
            for k, v in self.params.items():
                hf.create_dataset('param_%s' % k, data=v.asnumpy())
            for k, v in self.aux_params.items():
                hf.create_dataset('aux_param_%s' % k, data=v.asnumpy())

    def load(self, prefix):
        """Load model params from file.

        :param prefix: prefix of model name.
        """
        param_name = '%s.params' % prefix
        with h5py.File(param_name, 'r') as hf:
            for k, v in self.params.items():
                v[:] = numpy.array(hf.get('param_%s' % k))
            for k, v in self.aux_params.items():
                v[:] = numpy.array(hf.get('aux_param_%s' % k))

