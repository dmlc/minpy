import argparse
import random
import minpy.numpy as np
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter


class RNNNet(ModelBase):
    def __init__(self,
                 batch_size=100,
                 input_size=2,  # input dimension
                 hidden_size=64,
                 num_classes=1):
        super(RNNNet, self).__init__()
        self.add_param(name='h0', shape=(batch_size, hidden_size))\
            .add_param(name='Wx', shape=(input_size, hidden_size))\
            .add_param(name='Wh', shape=(hidden_size, hidden_size))\
            .add_param(name='b', shape=(hidden_size,))\
            .add_param(name='Wa', shape=(hidden_size, num_classes))\
            .add_param(name='ba', shape=(num_classes,))

    def forward(self, X):
        y1 = layers.rnn_forward(X, self.params['h0'], self.params['Wx'],
                                self.params['Wh'], self.params['b'])
        y2 = layers.affine(y1[:, -1, :], self.params['Wa'], self.params['ba'])
        return y2

    def loss(self, predict, y):
        assert predict is not None
        return layers.softmax_loss(predict, y)


def data_gen(N, seq_len=6, p=0.5, maxint=50):
    import numpy as np
    X_num = np.random.randint(0, high=maxint, size=(N, seq_len, 1))
    X_mask = np.ones((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in xrange(N):
        for j in xrange(seq_len):
            if random.random() > p:
                X_mask[i, j, 0] = 1
            else:
                X_mask[i, j, 0] = 0
        Y[i, 0] = np.sum(X_mask[i, :, :] * X_num[i, :, :])
    X = np.append(X_num, X_mask, axis=2) 
    return X, Y


def main(args=None):
    model = RNNNet()
    X_train, Y_train = data_gen(10000)       
    X_test, Y_test = data_gen(1000)

    train_dataiter = NDArrayIter(X_train,
                         Y_train,
                         batch_size=100,
                         shuffle=True)

    test_dataiter = NDArrayIter(X_test,
                         Y_test,
                         batch_size=100,
                         shuffle=False)

    solver = Solver(model,
                    train_dataiter,
                    test_dataiter,
                    num_epochs=10,
                    init_rule='xavier',
                    update_rule='sgd_momentum',
                    optim_config={
                        'learning_rate': 1e-4,
                        'momentum': 0.9
                    },
                    verbose=True,
                    print_every=20)
    solver.init()
    solver.train()


if __name__ == '__main__':
    main()
