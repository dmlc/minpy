class BatchNormalization(_Layer):
    def __init__(self, eps=1e-5, momentum=0.9, init_configs=None, update_configs=None):
        """ Batch normalization. To perform batch normalization on convolution layer outputs, please use SpatialBatchNormalization.
        param float epsilon: hyperparameter guaranteeing numeric stability. 1e-5 by default.
        param float momentum: hyperparameter controlling the speed at which running mean and running variance change.
        """
        self._eps = eps
        self._momentum = momentum
        params = ('gamma', 'beta')
        aux_param = ('running_mean', 'running_var')
        super(BatchNormalization, self).__init__()
        if init_configs is None:
            init_configs = {
                self.gamma: {'init_rule': 'constant', 'init_config': {'value': 1.0}},
                self.beta: {'init_rule': 'constant', 'init_config' : {'value' : 0.0}}
            }
        if update_configs is None:
            # TODO default
            update_configs = {
            }

    def forward(self, input, mode):
        outputs, running_mean, running_variance = layers.batchnorm(
            gamma, beta = self._get_params(self.gamma, self.beta)
            input, params[self.gamma], params[self.beta],
            params['__training_in_progress__'], self._epsilon, self._momentum,
            self.running_mean, self.running_variance)
        self.running_mean, self.running_variance = running_mean, running_variance
        return outputs

    def output_shape(self, input_shape):
        return input_shape

    def param_shapes(self, input_shape):
        return {self.gamma: input_shape, self.beta: input_shape}

    def parameter_settings(self):
        return 

class SpatialBatchNormalization(BatchNormalization):
    # pylint: disable=line-too-long
    count = 0

    def __init__(self, epsilon=1e-5, momentum=0.9):
        """ Spatial batch normalization of convolution layer outputs.

        param float epsilon: hyperparameter guaranteeing numeric stability. 1e-5 by default.
        param float momentum: hyperparameter controlling the speed at which running mean and running variance change.
        """
        super(SpatialBatchNormalization, self).__init__(
            epsilon=1e-5, momentum=0.9)
        self.gamma = 'SpatialBN%d_gamma' % self.__class__.count
        self.beta = 'SpatialBN%d_beta' % self.__class__.count

    def forward(self, input, params):
        N, C, W, H = input.shape
        input = transpose(input, (0, 2, 3, 1))
        input = np.reshape(input, (N * W * H, C))

        outputs, running_mean, running_variance = layers.batchnorm(
            input, params[self.gamma], params[self.beta],
            params['__training_in_progress__'], self.epsilon, self.momentum,
            self.running_mean, self.running_variance)
        self.running_mean, self.running_variance = running_mean, running_variance
        outputs = np.reshape(outputs, (N, W, H, C))
        outputs = transpose(outputs, (0, 3, 1, 2))
        return outputs

    def output_shape(self, input_shape):
        return input_shape

    def param_shapes(self, input_shape):
        return {self.gamma: (input_shape[0], ), self.beta: (input_shape[0], )}

    def parameter_settings(self):
        return {
            self.gamma: {
                'init_rule': 'constant',
                'init_config': {
                    'value': 1.0
                }
            },
            self.beta: {
                'init_rule': 'constant'
            }
        }

class Convolution(_Module):
    # pylint: disable=too-many-instance-attributes
    count = 0

    def __init__(self,
                 kernel_shape,
                 kernel_number,
                 stride=(1, 1),
                 pad=(0, 0),
                 initializer=None):
        # pylint: disable=too-many-arguments
        """ Convolution layer

        param tuple kernel_shape: the shape of kernel (x, y).
        param int kernel_number: the number of kernels.
        param tuple stride: stride (x, y).
        param tuple pad: padding.
        """

        super(Convolution, self).__init__()
        self.kernel_shape = kernel_shape
        self.kernel_number = kernel_number
        self.stride = stride
        self.pad = pad

        self.weight = 'convolution%d_weight' % self.__class__.count
        self.bias = 'convolution%d_bias' % self.__class__.count

        self.initializer = initializer

        self.__class__.count += 1

        self.input = mx.sym.Variable(name='input')
        self.convolution = mx.sym.Convolution(
            name='convolution',
            data=self.input,
            kernel=self.kernel_shape,
            num_filter=self.kernel_number,
            stride=self.stride,
            pad=self.pad)

    def forward(self, input, params):
        args = {
            'input': input,
            'convolution_weight': params[self.weight],
            'convolution_bias': params[self.bias]
        }
        return minpy.core.Function(self.convolution, {'input': input.shape})(
            **args)

    def output_shape(self, input_shape):
        _, output_shape, _ = self.convolution.infer_shape(
            input=tuple([1] + list(input_shape)))
        return map(int, output_shape[0][1:])

    def param_shapes(self, input_shape):
        assert len(input_shape) == 3, 'The input tensor should be 4D.'
        weight_shape = (self.kernel_number, input_shape[0],
                        self.kernel_shape[0], self.kernel_shape[1])  # pylint: disable=line-too-long
        bias_shape = (self.kernel_number, )
        return {self.weight: weight_shape, self.bias: bias_shape}

    def parameter_settings(self):
        return self.initializer if self.initializer else \
        {
            self.weight : {'init_rule' : 'xavier'},
            self.bias   : {'init_rule' : 'constant'}
        }


class Dropout(_Module):
    def __init__(self, p):
        """ Dropout layer

        param p: the probability at which the outputs of neurons are dropped.
        """

        super(Dropout, self).__init__()
        self.probability = p

    def forward(self, input, params):
        return layers.dropout(input, self.probability,
                              params['__training_in_progress__'])

    def output_shape(self, input_shape):
        return input_shape


class Pooling(_Module):
    count = 0

    def __init__(self, mode, kernel_shape, stride=(1, 1), pad=(0, 0)):
        """ Pooling layer
        param tuple kernel_shape: the shape of kernel (x, y).
        param tuple stride: stride (x, y).
        param tuple pad: padding.

        mode: 'avg', 'max', 'sum'
        """

        super(Pooling, self).__init__()
        self.kernel_shape = kernel_shape
        self.mode = mode
        self.stride = stride
        self.pad = pad

        self.__class__.count += 1

        self.input = mx.sym.Variable(name='input')
        self.pooling = mx.sym.Pooling(
            name='pooling',
            data=self.input,
            kernel=self.kernel_shape,
            pool_type=self.mode,
            stride=self.stride,
            pad=self.pad)

    def forward(self, input, params):
        args = {'input': input}
        return minpy.core.Function(self.pooling, {'input': input.shape})(
            **args)

    def output_shape(self, input_shape):
        _, output_shape, _ = self.pooling.infer_shape(
            input=tuple([1] + list(input_shape)))
        return map(int, output_shape[0][1:])


class Identity(_Module):
    def __init__(self):
        """ Identity transformation.
        """

        super(Identity, self).__init__()

    def forward(self, input, *args):
        return input

    def output_shape(self, input_shape):
        return input_shape

class ReLU(_Module):
    def __init__(self):
        """ Rectified linear unit.
        """

        super(ReLU, self).__init__()

    def forward(self, input, *args):
        return layers.relu(input)

    def output_shape(self, input_shape):
        return input_shape


class Sigmoid(_Module):
    def __init__(self):
        """ Sigmoid activation function.
        """

        super(Sigmoid, self).__init__()

    def forward(self, input, *args):
        return 1 / (1 + np.exp(-input))

    def output_shape(self, input_shape):
        return input_shape


class Tanh(_Module):
    def __init__(self):
        """ Hyperbolic tangent activation function.
        """

        super(Tanh, self).__init__()

    def forward(self, input, *args):
        return np.tanh(input)

    def output_shape(self, input_shape):
        return input_shape


class Reshape(_Module):
    def __init__(self, shape):
        """ Reshape the input.
        param tuple shape: the new shape of one sample, e.g. (3072,) or (3, 32, 32).
        """

        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input, *args):
        shape = tuple([input.shape[0]] + list(self.shape))
        return np.reshape(input, shape)

    def output_shape(self, input_shape):
        return self.shape


class Flatten(_Module):
    def __init__(self):
        """ Flatten the input.
        """

        super(Flatten, self).__init__()

    def forward(self, input, *args):
        shape = (input.shape[0], int(np.prod(np.array(input.shape[1:]))))
        return np.reshape(input, shape)

    def output_shape(self, input_shape):
        return (int(np.prod(np.array(input_shape))), )


