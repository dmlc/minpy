import minpy.nn.model_builder as builder

'''
Only supports static symbol and a friendly interface for layer customization for the time being.

A graph contains nodes and operators

In model_builder, an operator is a callable object.
First, user instantiate an operator: operator0 = Operator(...).
Then, user writes "node_b = operator0(node_a)", which creates an edge in computation graph:

node_a ----------> node_b
      operator0

This mechanism enables user to share operator easily.
For example, after writing "node_b = operator0(node_a)", if user writes "node_c = operator0(node_a)", 
then:

node_c <---------- node_a ----------> node_b
        operator0           operator0
'''



################################
# residual network (for CIFAR) #
################################

def _convolution(*args, **kwargs):
    '''
    builder.Convolution(*args, **kwargs) only specifies an operator (an edge in computation graph).
    builder.Sequential is responsible for organizing those edges into an sequential order.
    '''

    return builder.Sequential((
        builder.Convolution(*args, **kwargs),
        builder.ReLU(),
        builder.BatchNormalization(),
    ))

    '''
    builder.Sequential is only an operator as well. To complete computation graph, user must specify
    an input for the operator by calling the operator. However, we will not specify input to this operator 
    right now because this operator will later be used to compose more complex operators. After composition
    is finished, we can simply specify input to the operator resulting from composition instead of this one.

    If one would like to specify input immediately after operator definition:

    builder.Sequential((
        builder.Convolution(*args, **kwargs),
        builder.ReLU(),
        builder.BatchNormalization(),
    ))(input) # call the operator here
    '''

def _residual_module(filter_number, bottleneck=False):
    if bottleneck:
        # identity and residual are only operators
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
    return identity + residual # '+' creates another operator

# n controls total number of residual modules
# please refer to section 4.2 of "Deep Residual Learning for Image Recognition" for details
n = 3

# builder.Variable('data') is not an operator but a node that can be input of operator
network = builder.Variable('data')
network = _convolution(16, (3, 3), (1, 1), (1, 1))(network) # specify operator input right after operator definition

for filter_number in (16, 32):
    network = builder.Sequential(tuple(_residual_module(filter_number) for i in range(n)))(network)
    network = _residual_module(filter_number * 2, True)(network)

'''
builder.Sequential(tuple(_residual_module(filter_number) for i in range(n)))(network)

  is equivalent to

for i in range(n):
    network = _residual_module(filter_number)(network)

builder.Sequential provides an intuitive syntax of "stacking n layers".
'''

network = builder.Sequential(tuple(_residual_module(64) for i in range(n)))(network)

network = builder.Sequential((
    builder.Pooling(mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0)),
    builder.Flatten(),
    builder.FullyConnected(10),
))(network)

residual_network = builder.Model(network, loss='softmax_loss')



#############################################################################
# residual network involving weight sharing (illustration of layer sharing) #
#############################################################################

n = 3

network = builder.Variable('data')
network = _convolution(16, (3, 3), (1, 1), (1, 1))(network)

for filter_number in (16, 32):
    '''
    builder.Sequential requires a tuple of operators as input
    All references in the tuple received by builder.Sequential refer to one identical module,
    which enables operator-sharing.
    '''
    network = builder.Sequential((_residual_module(filter_number),) * n)(network)
    network = _residual_module(filter_number * 2, True)(network)

network = builder.Sequential((_residual_module(64),) * n)(network)

network = builder.Sequential((
    builder.Pooling(mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0)),
    builder.Flatten(),
    builder.FullyConnected(10),
))(network)

weight_sharing_residual_network = builder.Model(network, loss='softmax_loss')



##########################################
# unfolded rnn (illustration of slicing) #
##########################################

X_to_H = builder.FullyConnected(256, bias=None) # shared operator
H_to_H = builder.FullyConnected(256)            # shared operator
H_to_O = builder.FullyConnected(10)             # shared operator
activation = builder.Tanh()

N_STEPS = 8 # number of time steps

data = builder.Variable('data')
labels = builder.Variable('labels')

# temporal_data and temporal_labels are tuples of nodes, which are outputs of slicing operators.
temporal_data = builder.Slice(axis=1, output_number=N_STEPS)(data)
temporal_labels = builder.Slice(axis=1, output_number=N_STEPS)(labels)

total_loss = 0
H = 0
for data, labels in zip(temporal_data, temporal_labels):
    H = activation(X_to_H(data) + H_to_H(H))
    O = H_to_O(H)
    loss = builder.NLLLoss(O, labels)
    total_loss += loss

unfolded_rnn = builder.Model(total, loss=None) # set loss function to None since loss computation is integrated into network



##################################
# An example of customized layer #
##################################

class FullyConnected(builder.Layer):
    _module_name = 'fully_connected' # used to assign global name to parameter
    def __init__(self, n_hidden_units, init_configs=None, update_configs=None, name=None):
        """ Fully connected layer.

        param int n_hidden_units: number of hidden units.
        """

        # model_builder requires a unique name (global name) for every parameter.
        # For example, the weight of a fully-connected layer has a global name "fully_connected0_weight".
        # But one only needs to provide a simpler local name and model_builder will figure out the global one
        params = ('weight', 'bias')
        aux_params = None

        # register parameters
        super(FullyConnected, self).__init__(params, aux_params, name)

        # model_builder figures out global parameter name on the basis of module name and local parameter name
        # After registration, global parameter name can be referred to via object attribute.
        # For example, self.weight == 'fully_connected0_weight'.

        # Specify initializer and optimizer by global parameter name
        self._default_init_configs = {
            self.weight : builder.WEIGHT_INITIALIZER, # pre-defined initializer
            self.bias   : builder.BIAS_INITIALIZER,   # pre-defined initializer
        }
        self._default_update_configs = {'update_rule' : 'sgd', 'learning_rate' : 0.1}

        # routines
        self._register_init_configs(init_configs)     # method inherited from builder.Layer
        self._register_update_configs(update_configs) # method inherited from builder.Layer

        # layer customization starts here
        self._n_hidden_units = n_hidden_units
       
    def forward(self, input, params):
        from minpy.nn.layers import affine
        weight, bias = self._get_params(self.weight, self.bias) # get parameter using global name
        return affine(input, weight, bias)

    def output_shape(self, input_shape):
        # shape inference happens later
        N, D = input_shape
        return (N, self._n_hidden_units)

    def param_shapes(self, input_shape):
        # shape inference happens later
        N, D = input_shape
        return {self.weight : (D, self._n_hidden_units), self.bias : (self._n_hidden_units,)}
