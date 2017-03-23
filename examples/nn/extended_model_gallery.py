import minpy.nn.model_builder as builder

'''
Temporary strategy: only supports static symbol and a friendly interface for layer customization.
'''

# residual network (for CIFAR)
def _convolution(*args, **kwargs):
    '''
    builder.Convolution(*args, **kwargs) only specifies an operation (an edge in computation graph)
    builder.Sequential is responsible for organizing those edges into an sequential order
    '''
    return builder.Sequential((
        builder.Convolution(*args, **kwargs),
        builder.ReLU(),
        builder.BatchNormalization(),
    ))

def _residual_module(filter_number, bottleneck=False):
    if bottleneck:
        identity = _convolution(filter_number, (2, 2), (2, 2), (0, 0))
        residual = builder.Sequential((
            _convolution(filter_number // 4, (1, 1), (1, 1), (0, 0)),
            _convolution(filter_number // 4, (3, 3), (2, 2), (1, 1)),
            _convolution(filter_number, (1, 1), (1, 1), (0, 0)),
        ))
    else:
        identity = network
        residual = builder.Sequential((
            # positional arguments simplify layer declaration
            _convolution(filter_number, (3, 3), (1, 1), (1, 1)),
            # keyword arguments improve readability at the cost of simplicity
            _convolution(filter_number=filter_number, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1)),
        ))
    return identity + residual # '+' creates another symbol

# n controls total number of residual modules
# please refer to section 4.2 of "Deep Residual Learning for Image Recognition" for details
n = 3

network = builder.Variable('data')
network = _convolution(16, (3, 3), (1, 1), (1, 1))(network)

for filter_number in (16, 32):
    network = builder.Sequential(tuple(_residual_module(filter_number) for i in range(n)))(network)
    network = _residual_module(filter_number * 2, True)(network)

network = builder.Sequential(tuple(_residual_module(64) for i in range(n)))(network)

network = builder.Sequential((
    builder.Pooling(mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0)),
    builder.Flatten(),
    builder.FullyConnected(10),
))(network)

residual_network = builder.Model(network, loss='softmax_loss')

# residual network involving weight sharing (illustration of layer sharing)
n = 3

network = builder.Variable('data')
network = _convolution(16, (3, 3), (1, 1), (1, 1))(network)

for filter_number in (16, 32):
    # all references received by builder.Sequential refer to an identical module
    network = builder.Sequential((_residual_module(filter_number),) * n)(network)
    network = _residual_module(filter_number * 2, True)(network)

network = builder.Sequential((_residual_module(64),) * n)(network)

network = builder.Sequential((
    builder.Pooling(mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0)),
    builder.Flatten(),
    builder.FullyConnected(10),
))(network)

weight_sharing_residual_network = builder.Model(network, loss='softmax_loss')

# unfolded rnn (illustration of slicing)
X_to_H = builder.FullyConnected(256, bias=None)
H_to_H = builder.FullyConnected(256)
H_to_O = builder.FullyConnected(10)
activation = builder.Tanh()

N_STEPS = 8 # temporal length

data = builder.Variable('data')
labels = builder.Variable('labels')

# slice is a function returning a tuple of symbols (slicers)
temporal_data = builder.slice(data, axis=1, output_number=N_STEPS)
temporal_labels = builder.slice(labels, axis=1, output_number=N_STEPS)

total_loss = 0
H = 0
for data, labels in zip(temporal_data, temporal_labels):
    H = activation(X_to_H(data) + H_to_H(H))
    O = H_to_O(H)
    loss = builder.NLLLoss(O, labels)
    total_loss += loss

unfolded_rnn = builder.Model(total, loss=None) # set loss function to None since loss computation is integrated into network

class FullyConnected(builder.Layer):
    _module_name = 'fully_connected'
    def __init__(self, n_hidden_units, init_configs=None, update_configs=None, name=None):
        """ Fully connected layer.

        param int n_hidden_units: number of hidden units.
        """

        '''
        the global name of a parameter is an string that is unique to every parameter
        the local name of a parameter is used to refer to a parameter in a layer
        '''
        params = ('weight', 'bias') # "local" parameter name
        aux_params = None

        # register parameters
        # after registration, user could refer to global parameter name directly
        super(FullyConnected, self).__init__(params, aux_params, name)

        # register initializer and optimizer
        self._default_init_configs = {
            self.weight : builder.WEIGHT_INITIALIZER, # pre-defined initializer
            self.bias   : builder.BIAS_INITIALIZER,   # pre-defined initializer
        }
        self._default_update_configs = {'update_rule' : 'sgd', 'learning_rate' : 0.1}
        self._register_init_configs(init_configs)     # method inherited from builder.Layer
        self._register_update_configs(update_configs) # method inherited from builder.Layer

        # layer customization starts
        self._n_hidden_units = n_hidden_units
       
    def forward(self, input, params):
        from minpy.nn.layers import affine
        weight, bias = self._get_params(self.weight, self.bias)
        return affine(input, weight, bias)

    def output_shape(self, input_shape):
        # shape inference happens later
        N, D = input_shape
        return (N, self._n_hidden_units)

    def param_shapes(self, input_shape):
        # shape inference happens later
        N, D = input_shape
        return {self.weight : (D, self._n_hidden_units), self.bias : (self._n_hidden_units,)}
