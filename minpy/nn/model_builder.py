import minpy.numpy as np
import minpy.core
import minpy.nn.layers as layers
from minpy.nn.model import ModelBase
import mxnet as mx

class Module(object):
    """ The base class of building blocks of neural networks. """

    def __init__(self):
        return

    def parameter_shape(self, input_shape):
        """ Customized modules should override this function.

        :param tuple input_shape: The shape of one training sample, i.e. (3072,) or (3, 32, 32).
        :return: A dictionary mapping the names of parameters to their shapes.
        """

        return {}

    def parameter_settings(self):
        return {}


class Container(Module):
    def __init__(self):
        return


class Sequential(Container):
    def __init__(self, *args):
        """ Create feedforward networks. It is also used to construct sequential parts of networks.

        :param Module args: all layers of the feedforward networks in sequential order.
        """

        super(Sequential, self).__init__()
        if args:
            assert all(isinstance(arg, Module) for arg in args)
            self._modules = list(args)
        else:
            self._modules = []

    def forward(self, X, params):
        return reduce(
            lambda inputs, module : module.forward(inputs, params),
            self._modules,
            X
        )

    def output_shape(self, input_shape):
        return reduce(
            lambda shape, module : module.output_shape(shape),
            self._modules,
            input_shape
        )

    def parameter_shape(self, input_shape):
        shapes = {}
        def update_shapes(shape, module):
            shapes.update(module.parameter_shape(shape))
            return module.output_shape(shape)
        reduce(
            update_shapes,
            self._modules,
            input_shape
        )
        return shapes

    def parameter_settings(self):
        settings = {}
        for module in self._modules:
            settings.update(module.parameter_settings())
        return settings

    def append(self, module):
        """ Append a module (layer) at the end of the feedforward network.

        :param Module module: the module to append.
        :return: None.
        """

        assert isinstance(module, Module)
        self._modules.append(module)

    def insert(self, index, module):
        """ Insert a module (layer) at a specific position of the network.

        :param int index: the index at which the module (layer) is inserted.
        :param Module module: the module to insert.
        :return: None.
        """

        assert isinstance(module, Module)
        self._modules.insert(index, module)

    def remove(self, index):
        """ Remove the module at the specified index.

        :param int index: the index at which the module (layer) is removed.
        :return: None.
        """

        assert -1 < index and index < len(self._modules)
        del self._modules[index]


class Parallel(Container):
    def __init__(self):
        return


class Affine(Module):
    count = 0
    def __init__(self, hidden_number, initializer=None):
        """ Fully connected layer. (affine transformation)

        param int hidden_number: number of hidden units.
        """
        super(Affine, self).__init__()
        self._hidden_number = hidden_number 
        self.weight = 'fullyconnected%d_weight' % self.__class__.count
        self.bias   = 'fullyconnected%d_bias' % self.__class__.count

        self._initializer = initializer

        self.__class__.count += 1

    def forward(self, inputs, params):
        return layers.affine(inputs, params[self.weight], params[self.bias])

    def parameter_shape(self, input_shape):
        assert len(input_shape) == 1, 'Affine layer inputs must be 2D.'
        weight_shape = (input_shape[-1], self._hidden_number)
        bias_shape   = (self._hidden_number,)
        return {
            self.weight : weight_shape, 
            self.bias   : bias_shape 
        }

    def output_shape(self, input_shape):
        return (self._hidden_number,)
    
    def parameter_settings(self):
      return self._initializer if self._initializer else \
      {
          self.weight : {'init_rule' : 'xavier'},
          self.bias   : {'init_rule' : 'constant'}
      }
        

class BatchNormalization(Module):
    count = 0
    def __init__(self, epsilon=1e-5, momentum=0.9):
        """ Batch normalization. To perform batch normalization on convolution layer outputs, please use SpatialBatchNormalization.
        
        param float epsilon: hyperparameter guaranteeing numeric stability. 1e-5 by default.
        param float momentum: hyperparameter controlling the speed at which running mean and running variance change.
        """
        super(BatchNormalization, self).__init__()
        self.epsilon  = epsilon
        self.momentum = momentum
        self.running_mean, self.running_variance = None, None

        self.gamma = 'batchnorm%d_gamma' % self.__class__.count
        self.beta  = 'batchnorm%d_beta' % self.__class__.count

        self.__class__.count += 1

    def forward(self, inputs, params):
        outputs, running_mean, running_variance = layers.batchnorm(
            inputs,
            params[self.gamma],
            params[self.beta],
            params['__training_in_progress__'],
            self.epsilon,
            self.momentum,
            self.running_mean,
            self.running_variance
        )
        self.running_mean, self.running_variance = running_mean, running_variance
        return outputs

    def output_shape(self, input_shape):
        return input_shape

    def parameter_shape(self, input_shape):
        return {
            self.gamma : input_shape,
            self.beta  : input_shape
        }

    def parameter_settings(self):
        return {
            self.gamma : {'init_rule' : 'constant', 'init_config' : {'value': 1.0}},
            self.beta  : {'init_rule' : 'constant', 'init_config' : {}}
        }


class SpatialBatchNormalization(BatchNormalization):
    count = 0
    def __init__(self, epsilon=1e-5, momentum=0.9):
        """ Spatial batch normalization of convolution layer outputs.
        
        param float epsilon: hyperparameter guaranteeing numeric stability. 1e-5 by default.
        param float momentum: hyperparameter controlling the speed at which running mean and running variance change.
        """
        super(SpatialBatchNormalization, self).__init__(epsilon=1e-5, momentum=0.9)
        self.gamma = 'SpatialBN%d_gamma' % self.__class__.count
        self.beta  = 'SpatialBN%d_beta' % self.__class__.count

    def forward(self, inputs, params):
        N, C, W, H = inputs.shape
        inputs = _transpose(inputs, (0, 2, 3, 1))
        inputs = np.reshape(inputs, (N * W * H, C))

        outputs, running_mean, running_variance = layers.batchnorm(
            inputs,
            params[self.gamma],
            params[self.beta],
            params['__training_in_progress__'],
            self.epsilon,
            self.momentum,
            self.running_mean,
            self.running_variance
        )
        self.running_mean, self.running_variance = running_mean, running_variance
        outputs = np.reshape(outputs, (N, W, H, C))
        outputs = _transpose(outputs, (0, 3, 1, 2)) 
        return outputs

    def output_shape(self, input_shape):
        return input_shape

    def parameter_shape(self, input_shape):
        return {
            self.gamma : (input_shape[0],),
            self.beta  : (input_shape[0],)
        }

    def parameter_settings(self):
        return {
            self.gamma : {'init_rule' : 'constant', 'init_config' : {'value' : 1.0}},
            self.beta  : {'init_rule' : 'constant', 'init_config' : {}}
        }


class Convolution(Module):
    count = 0
    def __init__(self, kernel_shape, kernel_number, stride=(1, 1), pad=(0, 0), initializer=None):
        """ Convolution layer

        param tuple kernel_shape: the shape of kernel (x, y).
        param int kernel_number: the number of kernels.
        param tuple stride: stride (x, y).
        param tuple pad: padding.
        """

        super(Convolution, self).__init__()
        self._kernel_shape  = kernel_shape
        self._kernel_number = kernel_number
        self._stride        = stride
        self._pad           = pad

        self.weight = 'convolution%d_weight' % self.__class__.count
        self.bias   = 'convolution%d_bias' % self.__class__.count

        self._initializer = initializer

        self.__class__.count += 1

        inputs = mx.sym.Variable(name='inputs')
        self._convolution = mx.sym.Convolution(
            name       = 'convolution',
            data       = inputs,
            kernel     = self._kernel_shape,
            num_filter = self._kernel_number,
            stride     = self._stride,
            pad        = self._pad
        )

    def forward(self, inputs, params):
        args = {
            'inputs'             : inputs,
            'convolution_weight' : params[self.weight],
            'convolution_bias'   : params[self.bias]
        }
        return minpy.core.Function(self._convolution, {'inputs':inputs.shape})(**args)

    def output_shape(self, input_shape):
        _, output_shape, _ = self._convolution.infer_shape(inputs=tuple([1] + list(input_shape)))
        return _normal_shape(output_shape[0][1:])

    def parameter_shape(self, input_shape):
        assert len(input_shape) == 3, 'The input tensor should be 4D.'
        W, H = self._kernel_shape
        weight_shape = (self._kernel_number, input_shape[0], W, H)
        bias_shape   = (self._kernel_number,)
        return {
            self.weight : weight_shape, 
            self.bias   : bias_shape 
        }

    def parameter_settings(self):
      return self._initializer if self._initializer else \
          {
              self.weight : {'init_rule' : 'xavier'},
              self.bias   : {'init_rule' : 'constant'}
          }


class Dropout(Module):
    def __init__(self, p):
        """ Dropout layer
        param p: the probability at which the outputs of neurons are dropped.
        """
        super(Dropout, self).__init__()
        self._probability = p
    def forward(self, inputs, params):
        return layers.dropout(inputs, self._probability, params['__training_in_progress__'])
    def output_shape(self, input_shape):
        return input_shape
      

class Pooling(Module):
    count = 0
    def __init__(self, mode, kernel_shape, stride=(1, 1), pad=(0, 0)):
        """ Pooling layer
        param tuple kernel_shape: the shape of kernel (x, y).
        param tuple stride: stride (x, y).
        param tuple pad: padding.
        """

        '''
          mode: 'avg', 'max', 'sum'
        '''
        super(Pooling, self).__init__()
        self._kernel_shape  = kernel_shape
        self._mode          = mode
        self._stride        = stride
        self._pad           = pad

        self.__class__.count += 1

        inputs = mx.sym.Variable(name='inputs')
        self._pooling = mx.sym.Pooling(
            name       = 'pooling',
            data       = inputs,
            kernel     = self._kernel_shape,
            pool_type  = self._mode,
            stride     = self._stride,
            pad        = self._pad
        )

    def forward(self, inputs, params):
        args = {'inputs' : inputs}
        return minpy.core.Function(self._pooling, {'inputs':inputs.shape})(**args)

    def output_shape(self, input_shape):
        __, output_shape, __ = self._pooling.infer_shape(inputs=tuple([1] + list(input_shape)))
        return _normal_shape(output_shape[0][1:])


class Identity(Module):
    def __init__(self):
        """ Identity transformation.
        """

        super(Identity, self).__init__()
    def forward(self, inputs, *args):
        return inputs
    def output_shape(self, input_shape):
        return input_shape


class Export(Identity):
    def __init__(self, label, storage):
        super(Export, self).__init__()
        assert label not in storage, 'duplicated label'
        self._label, self._storage = label, storage
        self._storage[self._label] = None
    def forward(self, inputs, params):
        # TODO policy train_only test_only default
        self._storage[self._label] = inputs
        return inputs


class ReLU(Module):
    def __init__(self):
        """ Rectified linear unit.
        """

        super(ReLU, self).__init__()
    def forward(self, inputs, *args):
        return layers.relu(inputs)
    def output_shape(self, input_shape):
        return input_shape
   

class Sigmoid(Module):
    def __init__(self):
        """ Sigmoid activation function.
        """

        super(Sigmoid, self).__init__()
    def forward(self, inputs, *args):
        return 1 / (1 + np.exp(-inputs))
    def output_shape(self, input_shape):
        return input_shape


class Tanh(Module):
    def __init__(self):
        """ Hyperbolic tangent activation function.
        """
        super(Tanh, self).__init__()
    def forward(self, inputs, *args):
        return np.tanh(inputs)
    def output_shape(self, input_shape):
        return input_shape


class Reshape(Module):
    def __init__(self, shape):
        """ Reshape the inputs.
        param tuple shape: the new shape of one sample, e.g. (3072,) or (3, 32, 32).
        """
        super(Reshape, self).__init__()
        assert isinstance(shape, tuple)
        self._shape = shape
    def forward(self, inputs, *args):
        shape = (inputs.shape[0],) + self._shape
        return np.reshape(inputs, shape)
    def output_shape(self, input_shape):
        return self._shape


class Flatten(Module):
    def __init__(self):
        """ Flatten the inputs.
        """

        super(Flatten, self).__init__()
    def forward(self, inputs, *args):
        shape = (inputs.shape[0], int(np.prod(np.array(inputs.shape[1:]))))
        return np.reshape(inputs, shape)
    def output_shape(self, input_shape):
        return (int(np.prod(np.array(input_shape))),)


class Plus(Parallel):
    def __init__(self, left, right):
        """ Plus operator.
        param Module left: left operand.
        param Module right: right operand.
        """

        assert isinstance(left, Module)
        assert isinstance(right, Module)
        self._modules = [left, right]

    def forward(self, inputs, params):
        left, right = tuple(self._modules)
        left = left.forward(inputs, params)
        right = right.forward(inputs, params)
        return left + right

    def output_shape(self, input_shape):
        shape = self._modules[0].output_shape(input_shape)
        assert all(self._modules[i].output_shape(input_shape) == shape \
            for i in range(1, len(self._modules)))
        return shape

    def parameter_shape(self, input_shape):
        shapes = {}
        for module in self._modules:
            shapes.update(module.parameter_shape(input_shape))
        return shapes


class Model(ModelBase):
    """ """
    def __init__(self, container, loss, input_shape, data=None):
        """ Create the model from previous classes. The model is compatible with Minpy's solver.
        :param Module container: the network architecture.
        :param str or callable loss: The loss function. Options: 'l2', 'softmax', 'svm' or customized functions receivingpredict and label as parameters.
        :param input_shape: the shape of one sample, i.e. the shape of CIFAR-10 inputs is either (3072,), or (3, 32, 32).
        """

        self._loss_function = loss
        super(Model, self).__init__()
        self.__container = container
        
        if data is not None:
            self._initialize_layer_by_layer(container, data)   

        shapes = container.parameter_shape(input_shape)
        settings = container.parameter_settings()
        reduce(
            lambda arg, key : arg.add_param(name=key, shape=shapes[key], **settings[key]),
            shapes.keys(),
            self
        )

    def forward(self, X, mode):
        # TODO improve the method to distinguish training and test
        self.params['__training_in_progress__'] = mode

        outputs = self.__container.forward(X, self.params)
        del self.params['__training_in_progress__']
        return outputs

    def loss(self, prediction, labels):
        if self._loss_function in ('l2', 'softmax', 'svm'):
            return getattr(layers, '%s_loss' % self._loss_function)(prediction, labels)
        else:
            return self._loss_function(prediction, labels)

    def _initialize_layer_by_layer(self, container, inputs):
        input_shape = inputs.shape[1:]
        for module in container._modules:
            if isinstance(module, Container):
                # TODO handle parallel container
                inputs = self._initialize_layer_by_layer(module, inputs)

            else:
                shapes = module.parameter_shape(input_shape)
                input_shape = module.output_shape(input_shape)
                if isinstance(module, Affine) or isinstance(module, Convolution):
                    module._initializer = {}
                    for key, value in shapes.items():
                        if 'weight' in key:
                            E_X_2 = np.mean(inputs ** 2)
                            if isinstance(module, Affine):
                                n = value[0]
                            else:
                              C, W, H = value[1:]
                              n = C * W * H
                            std = 1 / (E_X_2 * n) ** 0.5
                            module._initializer[key] = {
                                'init_rule'   : 'gaussian',
                                'init_config' : {'stdvar' : std},
                            }
                        else:
                            # bias
                            module._initializer[key] = {
                                'init_rule'   : 'constant',
                                'init_config' : {},
                            }

                parameters = {}
                parameters['__training_in_progress__'] = 'train'
                parameter_settings = module.parameter_settings()
                for key, value in parameter_settings.items():
                    import minpy.nn.init as initializers
                    parameters[key] = getattr(
                        initializers,
                        value['init_rule']
                    )(
                        shapes[key],
                        value['init_config']
                    )
                if isinstance(container, Sequential):
                    inputs = module.forward(inputs, parameters)

        if isinstance(Container, Parallel):
            inputs = module.forward(inputs, parameters)

        return inputs


# assistance functions
def _normal_shape(shape):
    return tuple(int(d) for d in shape)


def _swapaxes(inputs, axis0, axis1):
    data = mx.sym.Variable(name='inputs')
    swapped = mx.sym.SwapAxis(data=data, dim1=axis0, dim2=axis1)
    return minpy.core.Function(swapped, {'inputs':inputs.shape})(inputs=inputs)


def _transpose(inputs, axes):
    data = mx.sym.Variable(name='inputs')
    transposed = mx.sym.transpose(data=data, axes=axes)
    return minpy.core.Function(transposed, {'inputs':inputs.shape})(inputs=inputs)
