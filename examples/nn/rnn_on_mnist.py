import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *


class RNNModel(Model):
    def __init__(self, hidden_state_size, activation='tanh'):
        super(RNNModel, self).__init__(loss='softmax_loss')

        self._rnn = RNN(hidden_state_size, activation)
        self._linear = FullyConnected(num_hidden=10)

    def forward(self, data, mode='training'):
        N, length, D = data.shape

        hidden = None

        for i in range(length):
            patch = data[:, i, :]
            hidden = self._rnn(patch, hidden)

        return self._linear(hidden)


class LSTMModel(Model):
    def __init__(self, hidden_state_size, activation='tanh'):
        super(LSTMModel, self).__init__(loss='softmax_loss')

        self._rnn = LSTM(hidden_state_size, activation)
        self._linear = FullyConnected(num_hidden=10)

    def forward(self, data, mode='training'):
        N, length, D = data.shape

        hidden, cell = None, None

        for i in range(length):
            patch = data[:, i, :]
            hidden, cell = self._rnn(patch, hidden, cell)

        return self._linear(hidden)


unpack_batch = lambda batch : (batch.data[0], batch.label[0])


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    # TODO LSTM does not work (due to an issue in core.Function)
    parser.add_argument('--rnn', type=str, default='RNN')
    args = parser.parse_args()

    from joblib import load
    data = load(args.data_dir + 'mnist.dat')
    sample_number = 50000
    train_data, test_data = data['train_data'][:sample_number], data['test_data'][:sample_number]

    eps = 1e-5
    train_data = (train_data - train_data.mean(axis=0)) / (train_data.std(axis=0) + eps)
    test_data = (test_data - test_data.mean(axis=0)) / (test_data.std(axis=0) + eps)

    N, D = train_data.shape
    patch_size = 7
    sequence_length = D / patch_size
    train_data = train_data.reshape((N, sequence_length, patch_size))

    N, _ = test_data.shape
    test_data = test_data.reshape((N, sequence_length, patch_size))

    from minpy.nn.io import NDArrayIter
    batch_size = 64
    train_data_iter = NDArrayIter(train_data, data['train_label'][:sample_number], batch_size, shuffle=True)
    test_data_iter = NDArrayIter(test_data, data['test_label'][:sample_number], batch_size, shuffle=False)

    if args.rnn == 'RNN': model = RNNModel(128)
    elif args.rnn == 'LSTM' : model = LSTMModel(128)

    updater = Updater(model, update_rule='rmsprop', learning_rate=0.002)
    
    iteration_number = 0
    for epoch_number in range(50):
        for iteration, batch in enumerate(train_data_iter):
            iteration_number += 1

            data, labels = unpack_batch(batch)
            grad_dict, loss = model.grad_and_loss(data, labels)
            updater(grad_dict)

            if iteration_number % 100 == 0:
                print 'iteration %d loss %f' % (iteration_number, loss)

        test_data_iter.reset()
        errors, samples = 0, 0
        for batch in test_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data, 'inference')
            predictions = np.argmax(scores, axis=1)
            errors += np.count_nonzero(predictions - labels)
            samples += data.shape[0]

        print 'epoch %d validation error %f' % (epoch_number, errors / float(samples))
