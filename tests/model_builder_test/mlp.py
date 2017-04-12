import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *


'''
class MLP(Model):
    def __init__(self, shapes, class_number):
        super(MLP, self).__init__(loss='softmax_loss')
        from mxnet.symbol import *
        network = Variable('data')
        for shape in shapes:
            network = FullyConnected(data=network, num_hidden=shape)
            network = Activation(data=network, act_type='relu')
        network = FullyConnected(data=network, num_hidden=class_number)
        self._symbol = Symbolic(network)

    def forward(self, data, mode='training'):
        if mode == 'training': self.training()
        elif mode == 'inference': self.inference()

        return self._symbol(data=data)
'''


class MLP(Model):
    def __init__(self, shapes, class_number):
        super(MLP, self).__init__(loss='softmax_loss')
        self._linears = tuple(FullyConnected(num_hidden=shape) for shape in shapes)
        self._normalizers = tuple(BatchNorm(fix_gamma=False) for shape in shapes)
        self._classifier = FullyConnected(num_hidden=class_number)

    def forward(self, data, mode='training'):
        if mode == 'training': self.training()
        elif mode == 'inference': self.inference()

        for linear, normalizer in zip(self._linears, self._normalizers):
            data = linear(data)
            data = normalizer(data)
            data = ReLU()(data)

        return self._classifier(data)



unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    from examples.utils.data_utils import get_CIFAR10_data
    data = get_CIFAR10_data(args.data_dir)
    
    from minpy.nn.io import NDArrayIter
    batch_size = 64
    train_data_iter = NDArrayIter(data=data['X_train'], label=data['y_train'], batch_size=batch_size, shuffle=True)
    val_data_iter = NDArrayIter(data=data['X_test'], label=data['y_test'], batch_size=batch_size, shuffle=False)
 
    from minpy.context import set_context, gpu
    set_context(gpu(args.gpu_index))

    model = MLP((3072,) * 10, 10)
    updater = Updater(model, update_rule='sgd', learning_rate=0.001, momentem=0.5)
    
    iteration_number = 0

    for epoch_number in range(50):
        # training
        train_data_iter.reset()
        for batch in train_data_iter:
            iteration_number += 1
            if iteration_number in (32000, 48000):
                updater.learning_rate = updater.learning_rate * 0.1
               
            data, labels = unpack_batch(batch)
            grad_dict, loss = model.grad_and_loss(data, labels) # compute gradient (execute model.forward implicitly)
            updater(grad_dict) # update parameters

            if iteration_number % 100 == 0:
                print 'iteration %d loss %f' % (iteration_number, loss)

        # validation
        val_data_iter.reset()
        errors, samples = 0, 0
        for batch in val_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data, True) # TODO training=False
            predictions = np.argmax(scores, axis=1)
            errors += np.count_nonzero(predictions - labels)
            samples += len(data)

        print 'epoch %d validation error rate %f' % (epoch_number, errors / float(samples))
