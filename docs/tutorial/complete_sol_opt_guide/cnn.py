import mxnet as mx

batch_size=128
input_size=(3, 32, 32)
flattened_input_size=3 * 32 * 32
hidden_size=512
num_classes=10

class ConvolutionNet(ModelBase):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
        # Define symbols that using convolution and max pooling to extract better features
        # from input image.
        net = mx.sym.Variable(name='X')
        net = mx.sym.Convolution(
                data=net, name='conv', kernel=(7, 7), num_filter=32)
        net = mx.sym.Activation(
                data=net, act_type='relu')
        net = mx.sym.Pooling(
                data=net, name='pool', pool_type='max', kernel=(2, 2),
                stride=(2, 2))
        net = mx.sym.Flatten(data=net)
        # Create forward function and add parameters to this model.
        self.conv = Function(
                net, input_shapes={'X': (batch_size,) + input_size},
                name='conv')
        self.add_params(self.conv.get_params())
        # Define ndarray parameters used for classification part.
        output_shape = self.conv.get_one_output_shape()
        conv_out_size = output_shape[1]
        self.add_param(name='w1', shape=(conv_out_size, hidden_size)) \
            .add_param(name='b1', shape=(hidden_size,)) \
            .add_param(name='w2', shape=(hidden_size, num_classes)) \
            .add_param(name='b2', shape=(num_classes,))

    def forward(self, X, mode):
        out = self.conv(X=X, **self.params)
        out = layers.affine(out, self.params['w1'], self.params['b1'])
        out = layers.relu(out)
        out = layers.affine(out, self.params['w2'], self.params['b2'])
        return out

    def loss(self, predict, y):
        return layers.softmax_loss(predict, y)
