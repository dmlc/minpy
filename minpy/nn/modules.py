import functools
import operator

import mxnet.symbol

import minpy.array
import minpy.numpy
import minpy.nn.layers
import minpy.nn.model_builder


class Identity(minpy.nn.model_builder.Layer):
    _module_name = 'identity'
    def __init__(self, name=None):
        super(Identity, self).__init__(name)

    def forward(self, X):
        return X


class FullyConnected(minpy.nn.model_builder.Layer):
    _module_name = 'fully_connected'
    def __init__(self, unit_number, init_configs=None, update_configs=None, name=None):
        """ Fully-connected layer.

        param int unit_number: number of hidden units.
        """
        params = ('weight', 'bias')
        aux_params = None
        super(FullyConnected, self).__init__(params, aux_params, name)

        self._register_init_configs(init_configs)
        self._register_update_configs(update_configs)

        self._unit_number = unit_number
       
    def forward(self, input):
        weight, bias = self._get_params(self.weight, self.bias)
        return minpy.numpy.dot(input, weight) + bias

    def param_shapes(self, input_shape):
        N, D = input_shape
        return {self.weight : (D, self._unit_number), self.bias : (self._unit_number,)}


class ReLU(minpy.nn.model_builder.Layer):
    _module_name = 'ReLU'
    def __init__(self, name=None):
        """ Rectified linear unit.
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, input, *args):
        return minpy.nn.layers.relu(input)


class Dropout(minpy.nn.model_builder.Layer):
    _module_name = 'dropout'
    def __init__(self, p):
        """ Dropout layer

        param p: the probability at which the outputs of neurons are dropped.
        """

        super(Dropout, self).__init__()
        self._p = p

    def forward(self, data):
        return layers.dropout(input, self._p)


class Logistic(minpy.nn.model_builder.Layer):
    def __init__(self):
        """ Logistic function.
        """

        super(Sigmoid, self).__init__()

    def forward(self, input, *args):
        return 1 / (1 + np.exp(-input))


class Tanh(minpy.nn.model_builder.Layer):
    def __init__(self):
        """ Hyperbolic tangent function.
        """

        super(Tanh, self).__init__()

    def forward(self, input, *args):
        return np.tanh(input)


class Reshape(minpy.nn.model_builder.Layer):
    def __init__(self, shape):
        """
        """

        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input, *args):
        return


class Flatten(minpy.nn.model_builder.Layer):
    _module_name = 'flatten'
    def __init__(self):
        """ Flatten.
        """

        super(Flatten, self).__init__()

    def forward(self, X):
        N = X.shape[0]
        D = functools.reduce(operator.mul, X.shape[1:], 1)
        return minpy.numpy.reshape(X, (N, D))


class Symbolic(minpy.nn.model_builder.Layer):
    # TODO help!
    _module_name = 'symbolic'
    def __init__(self, symbol, variables=None, init_configs=None, update_configs=None, name=None):
        self._symbol = symbol

        params = set(symbol.list_arguments())
        if variables is None: variables = ('data',)
        params = params.difference(set(variables))
        params = tuple(params)

        aux_params = tuple(symbol.list_auxiliary_states())
        super(Symbolic, self).__init__(params, aux_params, name)
        
    def forward(self, **kwargs):
        # TODO multiple outputs
        shapes = {key : value.shape for key, value in kwargs.items()}
        func = minpy.core.Function(self._symbol, shapes)
        return func(**kwargs)

    def param_shapes(self, **kwargs):
        arg_shapes, _, _ = self._symbol.infer_shape(**kwargs)
        return dict(zip(self._param_names, tuple(arg_shapes)))

    def aux_param_shapes(self, **kwargs):
        arg_shapes, _, _ = self._symbol.infer_shape(**kwargs)
        return dict(zip(self._aux_param_names, tuple(arg_shapes)))


class FullyConnected(Symbolic):
    def __init__(self, *args, **kwargs):
        name = kwargs.get('name', None)

        data = mxnet.symbol.Variable('data')
        fully_connected = mxnet.symbol.FullyConnected(data, *args, **kwargs)

        super(FullyConnected, self).__init__(fully_connected)

        init_configs = kwargs.get('init_configs', None)
        self._register_init_configs(init_configs)
        update_configs = kwargs.get('update_configs', None)
        self._register_update_configs(update_configs)

    def forward(self, data):
        return super(FullyConnected, self).forward(data=data)

    def param_shapes(self, input_shape):
        return super(FullyConnected, self).param_shapes(data=input_shape)

    def aux_param_shapes(self, input_shape):
        return {}


class Convolution(Symbolic):
    def __init__(self, *args, **kwargs):
        # TODO interface (currently mxnet)
        # TODO input weight/bias not supported currently
        # TODO name consistency with mxnet (for loading pre-trained mxnet model)

        name = kwargs.get('name', None)

        data = mxnet.symbol.Variable('data')
        convolution = mxnet.symbol.Convolution(data, *args, **kwargs)

        super(Convolution, self).__init__(convolution)

        init_configs = kwargs.get('init_configs', None)
        self._register_init_configs(init_configs)
        update_configs = kwargs.get('update_configs', None)
        self._register_update_configs(update_configs)

    def forward(self, data):
        return super(Convolution, self).forward(data=data)

    def param_shapes(self, input_shape):
        return super(Convolution, self).param_shapes(data=input_shape)

    def aux_param_shapes(self, input_shape):
        return {}


class Pooling(Symbolic):
    def __init__(self, *args, **kwargs):
        name = kwargs.get('name', None)

        data = mxnet.symbol.Variable('data')
        pooling = mxnet.symbol.Pooling(data, *args, **kwargs)

        super(Pooling, self).__init__(pooling)

    def forward(self, data):
        return super(Pooling, self).forward(data=data)

    def param_shapes(self, input_shape):
        return {}

    def aux_param_shapes(self, input_shape):
        return {}


class BatchNorm(Symbolic):
    def __init__(self, *args, **kwargs):
        # TODO training/inference
        # TODO interface (currently mxnet)

        name = kwargs.get('name', None)

        data = mxnet.symbol.Variable('data')
        batch_norm = mxnet.symbol.BatchNorm(data, *args, **kwargs)

        super(BatchNorm, self).__init__(batch_norm)

        # TODO merge into __init__
        init_configs = kwargs.get('init_configs', None)
        update_configs = kwargs.get('update_configs', None)

    def forward(self, data):
        return super(BatchNorm, self).forward(data=data)

    def param_shapes(self, input_shape):
        return super(BatchNorm, self).param_shapes(data=input_shape)

    def aux_param_shapes(self, input_shape):
        return super(BatchNorm, self).param_shapes(data=input_shape)
