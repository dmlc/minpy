import minpy.nn.model_builder as builder

'''
__init__: declare a module
__call__: connect (an)other symbol(s) to self, i.e. create an edge in graph
'''

# residual network (for CIFAR)
def _convolution(*args, **kwargs):
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
for d, l in zip(temporal_data, temporal_labels):
    H = activation(X_to_H(d) + H_to_H(data))
    O = H_to_O(H)
    loss = builder.NLLLoss(O, l)
    total_loss += loss

unfolded_rnn = builder.Model(total, loss=None) # set loss function to None since loss computation is integrated into network
