import mxnet.ndarray as _nd
import mxnet.symbol as _symbol

from minpy.nn.model_builder import Layer as _Layer


class _Operatorized(_Layer):
    _module_name = 'operatorized'

    def __init__(self, operator_name, **kwargs):
        init_configs = kwargs.pop('init_configs', None)
        update_configs = kwargs.pop('update_configs', None)
        name = kwargs.pop('name', None)

        variables = kwargs.pop('variables', None)
        self._variables = ('data',) if variables is None else variables
        
        self._kwargs = dict(kwargs)

        kwargs.update({v : _symbol.Variable(v) for v in self._variables})

        self._symbol = getattr(_symbol, operator_name)(name='symbol', **kwargs)
        self._operator = getattr(_nd, operator_name)

        self._module_name = operator_name.lower()

        eliminate_prefix = lambda name : name.replace('symbol_', '')

        params = set(self._symbol.list_arguments())
        params = params.difference(set(self._variables))
        params = tuple(name.replace('symbol_', '') for name in params)

        aux_params = tuple(self._symbol.list_auxiliary_states())
        aux_params = tuple(map(eliminate_prefix, aux_params))

        super(_Operatorized, self).__init__(params, aux_params, name)
    
    def forward(self, *args, **kwargs):
        is_array = lambda array : isinstance(array, _nd.NDArray)

        kwarg_dict = dict(zip(self._variables, filter(is_array, args))) 
        for key, value in kwargs.items():
            if is_array(value): kwarg_dict[key] = value
        
        kwarg_dict.update(dict(zip(self._module_param_names, self._get_params(*self._param_names))))
        kwarg_dict.update(dict(zip(self._module_aux_param_names, self._get_aux_params(*self._aux_param_names))))
        
        kwarg_dict.update(self._kwargs)

        return self._operator(**kwarg_dict)

   # TODO merge param_shapes and aux_param_shapes?
    def param_shapes(self, *args, **kwargs):
        kwargs.update(dict(zip(self._variables, args)))
        shapes, _, _ = self._symbol.infer_shape(**kwargs)

        shapes = dict(zip(self._symbol.list_arguments(), tuple(shapes)))
        for variable in self._variables:
            del shapes[variable]
        for param_name, shape in shapes.items():
            shapes[param_name.replace('symbol_', '')] = shapes.pop(param_name)
        local_to_global = dict(zip(self._module_param_names, self._param_names))
        shapes = {local_to_global[name] : shape for name, shape in shapes.items()}
        return shapes

    # TODO
    def aux_param_shapes(self, *args, **kwargs):
        kwargs.update(dict(zip(self._variables, args)))
        _, _, shapes = self._symbol.infer_shape(**kwargs)
        return dict(zip(self._aux_param_names, tuple(shapes)))


'''
criterion:
    1. an operator is parameterized
    2. an operator fits into frameworks of containers, especially Sequential
'''
Activation = lambda **kwargs : _Operatorized('Activation', **kwargs)
BatchNorm = lambda **kwargs : _Operatorized('BatchNorm', **kwargs)
Convolution = lambda **kwargs : _Operatorized('Convolution', **kwargs)
Deconvolution = lambda **kwargs : _Operatorized('Deconvolution', **kwargs)
Dropout = lambda **kwargs : _Operatorized('Dropout', **kwargs)
Embedding = lambda **kwargs : _Operatorized('Embedding', **kwargs)
FullyConnected = lambda **kwargs : _Operatorized('FullyConnected', **kwargs)
LeakyReLU = lambda **kwargs : _Operatorized('LeakyReLU', **kwargs)
Pooling = lambda **kwargs : _Operatorized('Pooling', **kwargs)
RNN = lambda **kwargs : _Operatorized('RNN', **kwargs)

# grammar sugar
ReLU = lambda : _Operatorized('Activation', act_type='relu')
Sigmoid = lambda : _Operatorized('Activation', act_type='sigmoid')
Tanh = lambda : _Operatorized('Activation', act_type='tanh')

RNNReLU = lambda **kwargs : _Operatorized('RNN', mode='rnn_relu', **kwargs)
RNNTanh = lambda **kwargs : _Operatorized('RNN', mode='rnn_tanh', **kwargs)
GRU = lambda **kwargs : _Operatorized('RNN', mode='gru', **kwargs)
LSTM = lambda **kwargs : _Operatorized('RNN', mode='lstm', **kwargs)

class Variable(_Layer):
    _module_name = 'variable'
    def __init__(self, shape, init_configs=None, update_configs=None, name=None):
        self._shape = shape
        params = ('variable',)
        aux_params = None
        super(Variable, self).__init__(params, aux_params, name)
        self._register_init_configs(init_configs)
        self._register_update_configs(update_configs)

    def forward(self):
        return self._get_param(self.variable)

    def param_shapes(self, *args):
        return {self.variable : self._shape}


class Identity(_Layer):
    _module_name = 'identity'
    def __init__(self, name=None):
        super(Identity, self).__init__(name)

    def forward(self, X):
        return X

class Reshape(_Layer):
    _module_name = 'reshape'
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self._shape = shape

    def forward(self, X, *args):
        return X.reshape(self._shape)


class BatchReshape(_Layer):
    _module_name = 'batch_reshape'
    def __init__(self, shape):
        super(BatchReshape, self).__init__()
        self._shape = shape

    def forward(self, X, *args):
        return X.reshape(X.shape[:1] + self._shape)


class Flatten(_Layer):
    _module_name = 'flatten'
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, X):
        return X.reshape((X.size,))


class BatchFlatten(_Layer):
    _module_name = 'flatten'
    def __init__(self):
        super(BatchFlatten, self).__init__()

    def forward(self, X):
        return X.reshape((X.shape[0], X[0].size,))


class FullyConnectedND(_Layer):
    _module_name = 'fullyconnected-nd'
    def __init__(self, num_hidden, name=None):
        super(FullyConnectedND, self).__init__(params=('weight', 'bias'))
        self._num_hidden = num_hidden
    def forward(self, X):
        if len(X.shape) > 2:
            X = _nd.flatten(X)
        return _nd.dot(X, self._model.params[self.weight]) + self._model.params[self.bias]
    def param_shapes(self, xshape):
        indim = 1
        for i in range(1, len(xshape)):
            indim *= xshape[i]
        return {self.weight : (indim, self._num_hidden), self.bias : (self._num_hidden,)}
