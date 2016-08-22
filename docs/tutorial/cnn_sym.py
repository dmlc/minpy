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
        net = mx.sym.FullyConnected(
                data=net, name='fc1', num_hidden=hidden_size)
        net = mx.sym.Activation(
                data=net, act_type='relu')
        net = mx.sym.FullyConnected(
                data=net, name='fc2', num_hidden=num_classes)
        net = mx.sym.SoftmaxOutput(
                data=net, name='output')
        # Create forward function and add parameters to this model.
        self.cnn = Function(
                net, input_shapes={'X': (batch_size,) + input_size},
                name='cnn')
        self.add_params(self.cnn.get_params())

    def forward(self, X, mode):
        out = self.cnn(X=X, **self.params)
        return out

    def loss(self, predict, y):
        return layers.softmax_cross_entropy(predict, y)
