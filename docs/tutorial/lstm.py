import layers
from minpy.nn.model import ModelBase


class LSTMNet(ModelBase):
    def __init__(self,
                 batch_size=200,
                 input_size=2,  # input dimension
                 hidden_size=100,
                 num_classes=1):
        super(LSTMNet, self).__init__()
        self.add_param(name='h0', shape=(batch_size, hidden_size))\
            .add_param(name='c0', shape=(batch_size, hidden_size))\
            .add_param(name='Wx', shape=(input_size, 4*hidden_size))\
            .add_param(name='Wh', shape=(hidden_size, 4*hidden_size))\
            .add_param(name='b', shape=(4*hidden_size,))\
            .add_param(name='Wa', shape=(hidden_size, num_classes))\
            .add_param(name='ba', shape=(num_classes,))

    def forward(self, X, mode):
        seq_len = X.shape[1]
        h = self.params['h0']
        c = self.params['c0']
        for t in xrange(seq_len):
            h, c = layers.lstm_step(X[:, t, :], h, c,
                                    self.params['Wx'],
                                    self.params['Wh'],
                                    self.params['b'])
        y = layers.affine(h, self.params['Wa'], self.params['ba'])
        return y

    def loss(self, predict, y):
        return layers.l2_loss(predict, y)
