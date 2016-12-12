""" Model base class codes. Adapted from cs231n lab codes. """
import h5py
import numpy
# pylint: disable=invalid-name

class ParamsNameNotFoundError(ValueError):
    """ Error of not existed name during accessing model params """
    pass

class UnknownAccessModeError(ValueError):
    """ Error of unexpected mode during accessing model params """
    pass

class ModelBase(object):
    """Base class for describing a neural network model."""
    def __init__(self):
        self.params = {}
        self.param_configs = {}
        self.aux_params = {}
        self.aux_param_configs = {}

    def add_param(self, name, shape, **kwargs):
        """ Add parameter.

        Parameters for training. This function
        will add variable "name" into dictionary self.param_config, which can be accessed by
        model in a solver.

        :param name: name of the param.
        :param shape: shape of the param.
        :param kwargs: contents of the param.
        :return: model itself
        """

        assert(name not in self.param_configs), 'Duplicate parameter name %s' % name
        self.param_configs[name] = {'shape': shape}
        self.param_configs[name].update(kwargs)
        return self

    def add_params(self, param_dict):
        """ Add parameter.

        Parameters for training. This function
        will add variable "name" into dictionary self.param_config, which can be accessed by
        model in a solver.

        :param param_dict: dictionary for param_name and value.
        :return: model itself
        """

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

    def forward_batch(self, batch, mode):
        """Do forward propagation.

        This is a more general interface than `forward` which only uses one input data.

        Parameters
        ---------
        batch
            A dictionary containing data/labels of the current batch.
            `batch.data` will return a list of arrays representing all the input data.
            `batch.label` will return a list of arrays representing all the input labels.
        mode
            A mode string that is either 'train' or 'test'.

        Returns
        ------
        Array
            Output of the model. TODO(minjie): support multiple outputs.
        """
        # Default implementation is to use only the first input data.
        return self.forward(batch.data[0], mode)

    def loss_batch(self, batch, forward_outputs):
        """Calculate the loss value of the current batch.

        This is a more general interface than `loss` which only uses one label.

        Parameters
        ---------
        batch
            A dictionary containing data/labels of the current batch.
            `batch.data` will return a list of arrays representing all the input data.
            `batch.label` will return a list of arrays representing all the input labels.
        forward_outputs
            An array representing the output of the model. TODO(minjie): support multiple outputs.

        Returns
        ------
        Value
            Loss value.
        """
        # Default implementation is to use only the first label.
        return self.loss(forward_outputs, batch.label[0])

    def forward(self, X, mode):
        """Do forward propagation.

        :param X: input vector
        :param mode: a mode string either 'train' or 'test'
        :return: output vector
        """
        raise NotImplementedError()

    def loss(self, predict, y):
        """Return the loss value given the output of the model.

        Parameters
        ----------
        predict
            An array representing the output of the model (return by `forward`).
        y
            Label of the current batch.

        Returns
        -------
        Value
            Loss value.
        """
        raise NotImplementedError()

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
