import minpy.numpy as np
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter
from examples.utils.data_utils import adding_problem_generator as data_gen


class RNNNet(ModelBase):
    def __init__(self,
                 batch_size=100,
                 input_size=2,  # input dimension
                 hidden_size=64,
                 num_classes=1):
        super(RNNNet, self).__init__()
        self.add_param(name='Wx', shape=(input_size, hidden_size))\
            .add_param(name='Wh', shape=(hidden_size, hidden_size))\
            .add_param(name='b', shape=(hidden_size,))\
            .add_param(name='Wa', shape=(hidden_size, num_classes))\
            .add_param(name='ba', shape=(num_classes,))

    def forward(self, X, mode):
        seq_len = X.shape[1]
        batch_size = X.shape[0]
        hidden_size = self.params['Wh'].shape[0]
        h = np.zeros((batch_size, hidden_size))
        for t in xrange(seq_len):
            h = layers.rnn_step(X[:, t, :], h, self.params['Wx'],
                                self.params['Wh'], self.params['b'])
        y = layers.affine(h, self.params['Wa'], self.params['ba'])
        return y

    def loss(self, predict, y):
        return layers.l2_loss(predict, y)


def main():
    model = RNNNet()
    x_train, y_train = data_gen(10000)
    x_test, y_test = data_gen(1000)

    train_dataiter = NDArrayIter(x_train,
                                 y_train,
                                 batch_size=100,
                                 shuffle=True)

    test_dataiter = NDArrayIter(x_test,
                                y_test,
                                batch_size=100,
                                shuffle=False)

    solver = Solver(model,
                    train_dataiter,
                    test_dataiter,
                    num_epochs=10,
                    init_rule='xavier',
                    update_rule='adam',
                    task_type='regression',
                    verbose=True,
                    print_every=20)
    solver.init()
    solver.train()


if __name__ == '__main__':
    main()
