batch_size=128
input_size=(3, 32, 32)
flattened_input_size=3 * 32 * 32
hidden_size=512
num_classes=10

class TwoLayerNet(ModelBase):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        # Define model parameters.
        self.add_param(name='w1', shape=(flattened_input_size, hidden_size)) \
            .add_param(name='b1', shape=(hidden_size,)) \
            .add_param(name='w2', shape=(hidden_size, num_classes)) \
            .add_param(name='b2', shape=(num_classes,)) \
            .add_param(name='gamma', shape=(hidden_size,),
                       init_rule='constant', init_config={'value': 1.0}) \
            .add_param(name='beta', shape=(hidden_size,), init_rule='constant') \
            .add_aux_param(name='running_mean', value=None) \
            .add_aux_param(name='running_var', value=None)

    def forward(self, X, mode):
        # Flatten the input data to matrix.
        X = np.reshape(X, (batch_size, 3 * 32 * 32))
        # First affine layer (fully-connected layer).
        y1 = layers.affine(X, self.params['w1'], self.params['b1'])
        # ReLU activation.
        y2 = layers.relu(y1)
        # Batch normalization
        y3, self.aux_params['running_mean'], self.aux_params['running_var'] = layers.batchnorm(
            y2, self.params['gamma'], self.params['beta'],
            running_mean=self.aux_params['running_mean'],
            running_var=self.aux_params['running_var'])
        # Second affine layer.
        y4 = layers.affine(y3, self.params['w2'], self.params['b2'])
        # Dropout
        y5 = layers.dropout(y4, 0.5, mode=mode)
        return y5

    def loss(self, predict, y):
        # ... Same as above
