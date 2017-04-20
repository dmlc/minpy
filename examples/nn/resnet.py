import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *


class ResNet(Model):
    def __init__(self, block_number, filter_numbers):
        super(ResNet, self).__init__(loss='softmax_loss')

        f0, f1, f2 = filter_numbers

        # register blocks reducing size of feature maps
        self._shrinking_blocks = (
            Convolution(
                num_filter = f0,
                kernel     = (3, 3),
                stride     = (1, 1),
                pad        = (1, 1),
                no_bias    = True,
            ),
            self._shrinking_block(f1),
            self._shrinking_block(f2),
        )

        # register blocks preserving size of feature maps
        self._residual_blocks = tuple(
            tuple(
                ResNet._residual_block(f) \
                  for i in range(block_number)
            ) for f in filter_numbers
        )

        # compute class scores from feature maps
        self._to_scores = Sequential(
            BatchNorm(fix_gamma=False),
            ReLU(),
            Pooling(pool_type='avg', kernel=(8, 8), stride=(1, 1)),
            BatchFlatten(),
            FullyConnected(num_hidden=10),
        )
    
    def forward(self, data, mode='training'):
        if mode == 'training': self.training()
        elif mode == 'inference': self.inference()

        data = data

        for shrink, residual in zip(self._shrinking_blocks, self._residual_blocks):
            data = shrink(data)
            for r in residual:
                data = data + r(data)

        return self._to_scores(data)

    @staticmethod
    def _convolution(**kwargs):
        defaults = {'kernel' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True}
        defaults.update(kwargs)
        return Sequential(
            BatchNorm(fix_gamma=False),
            ReLU(),
            Convolution(**defaults),
        )

    @staticmethod
    def _residual_block(f):
        residual = Sequential(
            ResNet._convolution(num_filter=f),
            ResNet._convolution(num_filter=f),
        )
        return residual

    @staticmethod
    def _shrinking_block(f):
        return Sequential(
            ResNet._convolution(num_filter=f, stride=(2, 2)),
            ResNet._convolution(num_filter=f),
        ) + ResNet._convolution(num_filter=f, kernel=(1, 1), stride=(2, 2), pad=(0, 0))


unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    '''
    from examples.utils.data_utils import get_CIFAR10_data
    data = get_CIFAR10_data(args.data_dir)

    from minpy.nn.io import NDArrayIter
    batch_size = 128
    train_data_iter = NDArrayIter(data=data['X_train'], label=data['y_train'], batch_size=batch_size, shuffle=True)
    val_data_iter = NDArrayIter(data=data['X_test'], label=data['y_test'], batch_size=batch_size, shuffle=False)
    '''

    from load_cifar10_data_iter import *
    train_data_iter, val_data_iter = load_cifar10_data_iter(batch_size=128, path=args.data_dir)

    from minpy.context import set_context, cpu, gpu
    if args.gpu_index < 0: set_context(cpu())
    else: set_context(gpu(args.gpu_index))

    model = ResNet(8, (16, 32, 64))
    updater = Updater(model, update_rule='sgd', learning_rate=0.1, momentem=0.9)
    
    epoch_number = 0
    iteration_number = 0
    terminated = False

    while not terminated:
        # training
        epoch_number += 1
        train_data_iter.reset()

        for iteration, batch in enumerate(train_data_iter):
            iteration_number += 1
            if iteration_number > 640000:
                terminated = True
                break
            if iteration_number in (32000, 48000):
                updater.learning_rate = updater.learning_rate * 0.1
               
            data, labels = unpack_batch(batch)
            loss = model(data, labels=labels)
            grad_dict = model.backward()
            updater(grad_dict)

            if iteration_number % 100 == 0:
                print 'iteration %d loss %f' % (iteration_number, loss)

        # validation
        '''
        val_data_iter.reset()
        errors, samples = 0, 0
        for batch in val_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data, 'inference')
            predictions = np.argmax(scores, axis=1)
            errors += np.count_nonzero(predictions - labels)
            samples += len(data)

        print 'epoch %d validation error %f' % (epoch_number, errors / float(samples))
        '''
