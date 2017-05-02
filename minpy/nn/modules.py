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
        params = tuple(map(eliminate_prefix, params))

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
globals()['Activation'] = lambda **kwargs : _Operatorized('Activation', **kwargs)
globals()['BatchNorm'] = lambda **kwargs : _Operatorized('BatchNorm', **kwargs)
globals()['Convolution'] = lambda **kwargs : _Operatorized('Convolution', **kwargs)
globals()['Deconvolution'] = lambda **kwargs : _Operatorized('Deconvolution', **kwargs)
globals()['Dropout'] = lambda **kwargs : _Operatorized('Dropout', **kwargs)
globals()['Embedding'] = lambda **kwargs : _Operatorized('Embedding', **kwargs)
globals()['FullyConnected'] = lambda **kwargs : _Operatorized('FullyConnected', **kwargs)
globals()['LeakyReLU'] = lambda **kwargs : _Operatorized('LeakyReLU', **kwargs)
globals()['Pooling'] = lambda **kwargs : _Operatorized('Pooling', **kwargs)

# grammar sugar
globals()['ReLU'] = lambda : _Operatorized('Activation', act_type='relu')
globals()['Sigmoid'] = lambda : _Operatorized('Activation', act_type='sigmoid')
globals()['Tanh'] = lambda : _Operatorized('Activation', act_type='tanh')


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


'''
class RNN(Symbolic):
    # TODO bidirectional
    def __init__(self, num_hidden, act_type):
        """ Perform one step of Elman RNN.
            param str act_type: 'relu', 'sigmoid', 'softrelu' or 'tanh'.
        """

        data = mxnet.symbol.Variable('data')
        hidden = mxnet.symbol.Variable('hidden')
        data = mxnet.symbol.FullyConnected(data=data, num_hidden=num_hidden, no_bias=True)
        hidden = mxnet.symbol.FullyConnected(data=hidden, num_hidden=num_hidden, name='HH')
        network = data + hidden
        network = mxnet.symbol.Activation(data=network, act_type=act_type)

        super(RNN, self).__init__(network, ('data', 'hidden'))

        self._register_init_configs({self.HH_weight : {'value' : minpy.numpy.eye(num_hidden)}})

        self._num_hidden = num_hidden
        
        # TODO default_init_configs
    
    def forward(self, data, hidden):
        if hidden is None:
            N, D = data.shape
            hidden = minpy.numpy.zeros((N, self._num_hidden))
        return super(RNN, self).forward(data=data, hidden=hidden)

    def param_shapes(self, data_shape, hidden_shape=None):
        if hidden_shape is None:
            N, D = data_shape
            hidden_shape = (N, self._num_hidden)
        return super(RNN, self).param_shapes(data=data_shape, hidden=hidden_shape)

    def aux_param_shapes(self, *args):
        return {}


class LSTM(Symbolic):
    def __init__(self, num_hidden, act_type):
        """ One step of LSTM.
        """
        data = mxnet.symbol.Variable('data')
        hidden = mxnet.symbol.Variable('hidden')
        data = mxnet.symbol.FullyConnected(data=data, num_hidden=4 * num_hidden, no_bias=True)
        hidden = mxnet.symbol.FullyConnected(data=hidden, num_hidden=4 * num_hidden)
        network = data + hidden

        sliced = mxnet.symbol.SliceChannel(data=network, num_outputs=4, axis=1)
        i = mxnet.symbol.Activation(data=sliced[0], act_type='sigmoid')
        f = mxnet.symbol.Activation(data=sliced[1], act_type='sigmoid')
        o = mxnet.symbol.Activation(data=sliced[2], act_type='sigmoid')
        g = mxnet.symbol.Activation(data=sliced[3], act_type='tanh')

        cell = mxnet.symbol.Variable('cell')
        next_cell = f * cell + i * g
        next_hidden = o * mxnet.symbol.Activation(data=next_cell, act_type=act_type)
        symbol = mxnet.symbol.Group((next_hidden, next_cell))

        super(LSTM, self).__init__(symbol, ('data', 'hidden', 'cell'))

        self._num_hidden = num_hidden

        # TODO default_init_configs
    
    def forward(self, data, hidden=None, cell=None):
        N, D = data.shape
        if hidden is None: hidden = minpy.numpy.zeros((N, self._num_hidden))
        if cell is None: cell = minpy.numpy.zeros((N, self._num_hidden))

        # returns next_hidden, next_cell
        return super(LSTM, self).forward(data=data, hidden=hidden, cell=cell)

    def param_shapes(self, data_shape, hidden_shape=None, cell_shape=None):
        if hidden_shape is None:
            N, D = data_shape
            hidden_shape = (N, self._num_hidden)

        if cell_shape is None:
            N, D = data_shape
            cell_shape = (N, self._num_hidden)

        return super(LSTM, self).param_shapes(data=data_shape, hidden=hidden_shape, cell=cell_shape)

    def aux_param_shapes(self, *args):
        return {}
'''
