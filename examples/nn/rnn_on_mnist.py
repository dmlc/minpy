import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *

class RNN(Model):
    def __init__(self, rnn, hidden_state_size, activation='tanh'):
        super(RNN, self).__init__()

        self._rnn = getattr(
            __import__('minpy.nn.modules'), rnn
        )(hidden_state_size, activation)

        self._linear = FullyConnected(num_hidden=class_number)

    def forward(self, data, mode='training'):
        if self._rnn == 'RNN': states = (None,)
        elif self._rnn == 'LSTM': states = (None, None)

        N, length, D = data.shape

        for i in range(length):
            patch = data[:, i, :]
            states = self._rnn(patch, *states)

        hidden_state = states[0]

        return self._linear(hidden_state)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    from joblib import load
    data = load(args.data_dir + 'mnist.dat')
    train_data, test_data = data['train_data'], data['test_data']
    print train_data.dtype
