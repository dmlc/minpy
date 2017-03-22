from functools import reduce as _reduce
import mxnet as mx
import minpy.numpy as np
from minpy.core import Function as _Function
import minpy.nn.layers as _layers
from minpy.nn.model import ModelBase as _ModelBase


_module_counter = {}


def _module_prefix(module_name):
    index = _module_counter.setdefault(module_name, 0)
    _module_counter[module_name] += 1
    prefix = '%s%d' % (module_name, index)
    return prefix


class _Module(object):
    # pylint: disable=too-few-public-methods
    """ Base class of module.
    :param dict initializer: Initializer.
    """

    _module_name = None
    def __init__(self, name):
        super(_Module, self).__init__()
        if self._module_name is None: raise NotImplementedError()
        self._name = _module_prefix(self._module_name) if name is None else name

    @property
    def name(self):
        return self._name

    def forward(self, data, mode):
        raise NotImplementedError()

    def output_shape(self, input_shape):
        raise NotImplementedError()

    def __setitem__(self, _):
        raise NotImplementedError()

    def __setitem__(self, _):
        raise NotImplementedError()


class _Container(_Module):
    def __init__(self, name):
        super(_Container, self).__init__(name)


class Sequential(_Container):
    _module_name = 'sequential'
    def __init__(self, *args, **kwargs):
        """ Sequential network.

        :param _Module args: all layers of the feedforward networks in sequential order.
        """

        name = kwargs.get('name', None)
        super(Sequential, self).__init__(name)
        assert all(isinstance(arg, _Module) for arg in args), TypeError()
        self._modules = list(args)

        self.append = self._modules.append
        self.insert = self._modules.insert
        self.pop = self._modules.pop
        self.reverse = self._modules.reverse

    def __iter__(self):
        for module in self._modules:
            yield module

    def forward(self, data, mode):
        forward_module = lambda input, module : module.forward(input, mode)
        return _reduce(forward_module, self._modules, data)

    def output_shape(self, input_shape):
        module_output_shape = lambda input_shape, module : module.output_shape(input_shape)
        return _reduce(module_output_shape, self._modules, input_shape)

    def param_shapes(self, input_shape):
        shapes = {}
        for module in self._modules:
            shapes.update(module.param_shapes(input_shape))
            input_shape = module.output_shape(input_shape)
        return shapes

    def aux_param_shapes(self, input_shape):
        shapes = {}
        for module in self._modules:
            shapes.update(module.aux_param_shapes(input_shape))
            input_shape = module.output_shape(input_shape)
        return shapes


def _register_configs(configs, to_register):
    # used in _Layer and _ConfigParser
    # register global attributes
    local_configs = {}
    for identifier, attr_value in to_register.items():
        if isinstance(identifier, str):
            if identifier in configs: local_configs[identifier] = attr_value
            else: 
                for config in configs.values(): config[identifier] = attr_value
        else: raise Exception()
    # register parameter-specific attributes
    for identifier, attr_value in local_configs.items():
          configs[identifier].update(attr_value)


class _Layer(_Module):
    def __init__(self, params=None, aux_params=None, name=None):
        super(_Layer, self).__init__(name)

        if params is None: params = tuple()
        self._module_param_names = params
        self._param_names = self._assign_param_names(*params)
        for param, param_name in zip(params, self._param_names):
            setattr(self, param, param_name)
        self._model_params = None

        if aux_params is None: aux_params = tuple()
        self._module_aux_param_names = aux_params
        self._aux_param_names = self._assign_param_names(*aux_params)
        for aux_param, aux_param_name in zip(aux_params, self._aux_param_names):
            setattr(self, '_%s' % aux_param, aux_param_name)
        self._model_aux_params = None

    def __getitem__(self, layer_name):
        if layer_name == self._name: return self
        else: raise KeyError()

    def _assign_param_names(self, *params):
        return tuple('%s_%s' % (self._name, param) for param in params)

    def _get_params(self, *param_names):
        try: 
            if len(param_names) > 1:
                return tuple(self._model_params[param_name] for param_name in param_names)
            else: return self._model_params[param_name]
        except KeyError: raise Exception('Parameters not initialized.')

    def _get_aux_param(self, aux_param_name):
        try: return self._model_aux_params[aux_param_name]
        except: raise Exception('Auxiliary parameters not initialized.')
    
    def _parse_param_configs(self, configs):
        if configs is None: return {}
        _configs = {key : value for key, value in configs.items()}
        for identifier in _configs:
            if isinstance(identifier, tuple):
                config = _configs.pop(identifier)
                for param_name in identifier: _configs[param_name] = config
        for module_param_name, param_name in zip(self._module_param_names, self._param_names):
            if module_param_name in _configs:
                _configs[param_name] = _configs.pop(module_param_name)
        for module_aux_param_name, aux_param_name in zip(self._module_aux_param_names, self._aux_param_names):
            if module_aux_param_name in _configs:
                _configs[aux_param_name] = _configs.pop(module_aux_param_name)
        return _configs

    def _register_init_configs(self, init_configs):
        self._init_configs = {param_name : {} for param_name in self._param_names}

        default_configs = self._parse_param_configs(self._default_init_configs)
        _register_configs(self._init_configs, default_configs)

        init_configs = self._parse_param_configs(init_configs)
        _register_configs(self._init_configs, init_configs)

    def _register_update_configs(self, update_configs):
        self._update_configs = {param_name : {} for param_name in self._param_names}

        default_configs = self._parse_param_configs(self._default_update_configs)
        _register_configs(self._update_configs, default_configs)

        update_configs = self._parse_param_configs(update_configs)
        _register_configs(self._update_configs, update_configs)

    @property
    def params(self):
        return dict(zip(self._param_names, self._get_params(*self._param_names)))

    @property
    def aux_params(self):
        return {aux_param_name : self._get_aux_param(aux_param_name) for aux_param_name in self._aux_param_names}
    
    def param_shapes(self, input_shape):
        return {}

    def aux_param_shapes(self, input_shape):
        return {}


class FullyConnected(_Layer):
    _module_name = 'fully_connected'
    def __init__(self, n_hidden_units, init_configs=None, update_configs=None, name=None):
        """ Fully connected layer.

        param int n_hidden_units: number of hidden units.
        """
        params = ('weight', 'bias')
        aux_params = None
        super(FullyConnected, self).__init__(params, aux_params, name)
        self._default_init_configs = {
            self.weight : {'init_rule' : 'xavier'},
            self.bias   : {'init_rule' : 'constant', 'value' : 0}
        }
        self._default_update_configs = {
            'update_rule'   : 'sgd',
            'learning_rate' : 0.1
        }
        self._register_init_configs(init_configs)
        self._register_update_configs(update_configs)
        self._n_hidden_units = n_hidden_units
       
    def forward(self, input, params):
        weight, bias = self._get_params(self.weight, self.bias)
        return _layers.affine(input, weight, bias)

    def output_shape(self, input_shape):
        N, D = input_shape
        return (N, self._n_hidden_units)

    def param_shapes(self, input_shape):
        N, D = input_shape
        return {self.weight : (D, self._n_hidden_units), self.bias : (self._n_hidden_units,)}


class ReLU(_Layer):
    _module_name = 'ReLU'
    def __init__(self, name=None):
        """ Rectified linear unit.
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, input, *args):
        return _layers.relu(input)

    def output_shape(self, input_shape):
        return input_shape


class Model(_ModelBase):
    def __init__(self, module, input_shape, loss_func=None):
        """ Create model from composition of modules.
        :param _Module module: network
        :param input_shape: shape of one sample, e.g. shape of CIFAR-10 data might be (3072,) or (3, 32, 32).
        """

        super(Model, self).__init__()
        self._module = module

        for param_name, shape in module.param_shapes(input_shape).items():
            self.add_param(param_name, shape)
        for aux_param_name, shape in module.aux_param_shapes(input_shape).items():
            self.add_aux_param(aux_param_name, shape)

        self._submodules = {submodule.name : submodule for submodule in self._module_iter()}

        self._init_configs = {}
        self._update_configs = {}
        for submodule in self._submodules.values():
            try:
                self._init_configs.update(submodule._init_configs)
                self._update_configs.update(submodule._update_configs)
            except AttributeError: pass
        print self._init_configs

        if isinstance(loss_func, str):
            self.loss_func = getattr(_layers, loss_func)
        else:
            self.loss_func = loss_func

    def __getitem__(self, module_name):
        return self._submodules[module_name]

    def _module_iter(self, module=None):
        if module is None: module = self._module
        for submodule in module:
            yield submodule
            if isinstance(submodule, _Container):
                for sub_submodule in self._module_iter(submodule):
                    yield sub_submodule

    def forward(self, data, mode, *args):
        return self._module.forward(data, mode)

    def loss(self, data, labels, *args):
        if self.loss_func is None: raise Exception()
        predictions = self.forward(data, 'train')
        return self.loss_func(predictions, labels)
   
    # TODO hook
    def backward(self, data, labels):
        from minpy.core import grad

        def loss(*args):
            if self.loss_func is None: raise Exception()
            predictions = self.forward(data, 'train')
            return self.loss_func(predictions, labels)
     
        param_names = tuple(self.params)
        params = tuple(self.params.values())
        grad_func = grad(loss, range(len(params)))
        grad_list = grad_func(*params)
        grad_dict = dict(zip(param_names, grad_list))

        return grad_dict


class _ConfigParser(object):
    class _AttrRef(object):
        def __init__(self, configs, attr):
            self._configs = configs
            self._attr = attr

        # TODO not float(self)
        def __add__(self, other):
            return float(self) + float(other)

        def __sub__(self, other):
            return float(self) - float(other)

        def __mul__(self, other):
            return float(self) * float(other)

        # TODO div family
        def __div__(self, other):
            return float(self) + float(other)

        def __iadd__(self, other):
            for config in self._configs:
                try: config[self._attr] += other
                except KeyError: pass

        def __isub__(self, other):
            for config in self._configs:
                try: config[self._attr] -= other
                except KeyError: pass

        def __imul__(self, other):
            for config in self._configs:
                try: config[self._attr] *= other
                except KeyError: pass

        # TODO idiv family
        def __idiv__(self, other):
            return float(self) + float(other)

        def __getitem__(self, param_name):
            return self._configs[param_name][self._attr]

        def __setitem__(self, param_name, attr_value):
            self._configs[param_name][self._attr] = attr_value

        def __float__(self):
            return float(self._attr_value)

        def __str__(self):
            return str(self._attr_value)

        def __repr__(self):
            return str(self._attr_value)

        @property
        def _attr_value(self):
            attr_values = set(config[self._attr] for config in self._configs.values() if self._attr in config)
            assert len(attr_values) is 1, Exception('Inconsistent or non-existent attribute.')
            return tuple(attr_values)[0]

    class _ParamRef(object):
        def _get(self, attr):
            return self.__getattribute__(attr)

        def _set(self, attr, attr_value):
            object.__setattr__(self, attr, attr_value)

        def __init__(self, configs, param_name):
            self._set('_configs', configs)
            self._set('_param_name', param_name)

        def __getattr__(self, attr):
            return self._get('_configs')[self._param_name][attr]

        def __setattr__(self, attr, attr_value):
            self._get('_configs')[self._param_name][attr] = attr_value

    def _get(self, attr):
        return self.__getattribute__(attr)

    def _set(self, attr, attr_value):
        object.__setattr__(self, attr, attr_value)

    def __init__(self, configs):
        super(_ConfigParser, self).__setattr__('_configs', configs)
    
    def __getattr__(self, attr):
        return _ConfigParser._AttrRef(self._configs, attr)

    def __setattr__(self, attr, attr_value):
        for config in self._configs.values():
            config[attr] = attr_value

    def __getitem__(self, param_name):
        return _ConfigParser._ParamRef(self._configs, param_name)

    def __setitem__(self, param_name, attr_values):
        self._configs[param_name] = attr_values

class Initializer(_ConfigParser):
    def __init__(self, model):
        super(Initializer, self).__init__(model._init_configs)
        self._set('_model', model)

    def __call__(self):
        import minpy.nn.init as init

        self._model.params = {}
        for param_name, config in self._get('_model').param_configs.items():
            init_config = self._get('_model')._init_configs[param_name]
            param_shape = config['shape']
            self._get('_model').params[param_name] = \
                getattr(init, init_config['init_rule'])(param_shape, init_config)

        self._model.aux_params = {}
        for aux_param_name, config in self._get('_model').aux_param_configs:
            init_config = self._get('_model')._init_configs[aux_param_name]
            aux_param_shape = config['shape']
            self._get('_model').aux_params[aux_param_name] = \
                getattr(init, init_config['init_rule'])(aux_param_shape, init_config)

        for module in self._get('_model')._module_iter():
            module._model_params = self._get('_model').params
            module._model_aux_params = self._get('_model').aux_params

class Updater(_ConfigParser):
    def __init__(self, model):
        super(Updater, self).__init__(model._update_configs)
        self._set('_model', model)
    
    def __call__(self, grad_dict):
        """ Only update parameters corresponding to gradients contained in grad_dict.
            User could update parameters selectively by manipulating grad_dict.
        """
        import minpy.nn.optim as optim
        for param_name, grad in grad_dict.items():
            param = self._get('_model').params[param_name]
            update_rule = self._get('_model')._update_configs[param_name]['update_rule']
            update_config = self._get('_model')._update_configs[param_name]
            self._get('_model').params[param_name], _update_config = \
                getattr(optim, update_rule)(param, grad, update_config)
            update_config.update(_update_config)
