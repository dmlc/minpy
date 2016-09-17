weight_decay = 0.001

class ConvolutionNet(ModelBase):
    def __init__(self):
        # ... Same as above.

    def forward(self, X, mode):
        # ... Same as above.

    def loss(self, predict, y):
        # Add L2 regularization for all the weights.
        reg_loss = 0.0
        for name, weight in self.params.items():
            reg_loss += np.sum(weight ** 2) * 0.5
        return layers.softmax_cross_entropy(predict, y) + weight_decay * reg_loss
