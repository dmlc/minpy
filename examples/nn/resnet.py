"""
Hyper-parameters are identical to those detailed in "Deep Residual Learning for Image Recognition".
"""


import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *


class ResNet(Model):
    def __init__(self, block_number, filter_numbers):
        super(ResNet, self).__init__(loss='softmax_loss')

        f0, f1, f2 = filter_numbers

        # register blocks reducing size of feature maps
        self._shrinking_blocks = (
            Sequential(
                BatchNorm(fix_gamma=False),
                self._convolution(num_filter=f0),
            ),
            self._shrinking_block(f1),
            Sequential(
                self._shrinking_block(f2),
                BatchNorm(fix_gamma=False),
                ReLU(),
            )
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
            Sequential(
                ResNet._convolution(num_filter=f, stride=(2, 2)),
                ResNet._convolution(num_filter=f),
            ) + \
            ResNet._convolution(num_filter=f, kernel=(1, 1), stride=(2, 2), pad=(0, 0))
        )


unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    from load_cifar10_data_iter import *
    train_data_iter, val_data_iter = load_cifar10_data_iter(batch_size=256, path=args.data_dir)

    from minpy.context import set_context, gpu
    set_context(gpu(args.gpu_index))

    model = ResNet(3, (16, 32, 64))
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
            if iteration_number > 64000:
                terminated = True
                break
            if iteration_number in (32000, 48000):
                updater.learning_rate = updater.learning_rate * 0.1
               
            data, labels = unpack_batch(batch)
            grad_dict, loss = model.grad_and_loss(data, labels)
            updater(grad_dict)

            if iteration_number % 100 == 0:
                print 'iteration %d loss %f' % (iteration_number, loss)

            try: open('log', 'w').close()
            except: pass
        
        # validation
        val_data_iter.reset()
        errors, samples = 0, 0
        for batch in val_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data, False) # TODO training=False
            predictions = np.argmax(scores, axis=1)
            errors += np.count_nonzero(predictions - labels)
            samples += len(data)

        print 'epoch %d validation error %f' % (epoch_number, errors / float(samples))
