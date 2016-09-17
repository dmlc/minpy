'''
  Minpy model builder example.

  For details about how to train a model with solver, please refer to:
    http://minpy.readthedocs.io/en/latest/tutorial/complete.html

  More models are available in minpy.nn.model_gallery.
'''

#... other import modules
import minpy.nn.model_builder as builder

class TwoLayerNet(ModelBase):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        # Define model parameters.
        self.add_param(name='w1', shape=(flattened_input_size, hidden_size)) \
            .add_param(name='b1', shape=(hidden_size,)) \
            .add_param(name='w2', shape=(hidden_size, num_classes)) \
            .add_param(name='b2', shape=(num_classes,))

    def forward(self, X, mode):
        # Flatten the input data to matrix.
        X = np.reshape(X, (batch_size, 3 * 32 * 32))
        # First affine layer (fully-connected layer).
        y1 = layers.affine(X, self.params['w1'], self.params['b1'])
        # ReLU activation.
        y2 = layers.relu(y1)
        # Second affine layer.
        y3 = layers.affine(y2, self.params['w2'], self.params['b2'])
        return y3

    def loss(self, predict, y):
        # Compute softmax loss between the output and the label.
        return layers.softmax_loss(predict, y)

def main(args):
    # Define a 2-layer perceptron
    MLP = builder.Sequential(
        builder.Affine(512),
        builder.ReLU(),
        builder.Affine(10)
    )

    # Cast the definition to a model compatible with minpy solver
    model = builder.Model(MLP, 'softmax', (3 * 32 * 32,))

    # ....
    # the above is equivalent to use the TwoLayerNet, as below (uncomment )
    # model = TwoLayerNet()

    solver = Solver(model,
    # ...
