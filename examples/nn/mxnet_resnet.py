from mxnet.symbol import *

def _convolution(**kwargs):
    return Convolution(no_bias=True, cudnn_tune='limited_workspace', **kwargs)

def _module(network, filter_number, shrink=False):
    # TODO shrink
    if shrink: identity = \
    _convolution(data=network, num_filter=filter_number, kernel=(1, 1), stride=(2, 2), pad=(0, 0))
    else: identity = network

    residual = BatchNorm(data=network, fix_gamma=False)
    residual = Activation(data=residual, act_type='relu')
    stride = (2, 2) if shrink else (1, 1)
    residual = ._convolution(data=residual, num_filter=filter_number, kernel=(3, 3), stride=stride, pad=(1, 1))
    residual = BatchNorm(data=residual, fix_gamma=False)
    residual = Activation(data=residual, act_type='relu')
    residual = _convolution(data=residual, num_filter=filter_number, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    return identity + residual


network = Variable('data')
network = _convolution(data=network, num_filter=16, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

for filter_number in (16, 32):
for i in range(block_number): network = _module(network, filter_number)
network = _module(network, filter_number * 2, shrink=True)

for i in range(block_number): network = _module(network, 64)

network = BatchNorm(data=network, fix_gamma=False)

network = Pooling(data=network, pool_type='avg', kernel=(8, 8), stride=(1, 1), pad=(0, 0))
network = Flatten(data=network)
network = FullyConnected(data=network, num_hidden=10)
network = SoftmaxOutput(data=network, normalization='batch')

from functools import reduce

args = network.list_arguments()
aux_states = network.list_auxiliary_states()

batch_size = 64
arg_shapes, _, aux_state_shapes = network.infer_shape(data=(batch_size, 3, 32, 32))

context = mx.gpu(settings.gpu_index)

arg_dict = {arg : mx.nd.zeros(context) for arg, shape in zip(args, arg_shapes)}
for arg, array in arg_dict.items():
    if 'weight' in arg:
        shape = array.shape
        if len(shape) == 4:
           fan_in = shape[1] * shape[2] * shape[3] 
        else: fan_in = 0
        fan_out = shape[0]
        d = (6.0 / (fan_in + fan_out)) ** 0.5
        arg_dict[arg][:] = mx.nd.uniform(low=-d, high=d, shape=shape)
    elif 'bias' in arg:
        arg_dict[arg][:] = 0
    elif 'gamma' in arg:
        arg_dict[arg][:] = 1
    elif 'beta' in arg:
        arg_dict[arg][:] = 0

args_grad = {
    arg : mx.nd.zeros(context) for arg, shape in zip(args, arg_shapes) \
        if arg not in ('data', 'softmax_label')
}

cache_dict = {
    arg : mx.nd.zeros(context) for arg, shape in zip(args, arg_shapes) \
        if arg not in ('data', 'softmax_label')
}

aux_state_dict = {aux_state : mx.nd.zeros(context) for aux_state, shape in zip(aux_states, aux_state_shapes)}
for aux_state, array in aux_state_dict.items():
    if 'mean' in aux_state:
        aux_state_dict[aux_state][:] = 0
    elif 'var' in aux_state:
        aux_state_dict[aux_state][:] = 1

executor = network.bind(context, args_dict, args_grad=args_grad, aux_state=aux_state_dict)


def momentem_update(param, grad, V, lr, momentem):
    V[:] = momentem * V - lr * grad
    param += V
    

if __name__ == '__main__':
    import time # TODO further profiling

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--gpu_index', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=False)
    args = parser.parse_args()

    from load_cifar10_data_iter import *
    train_data_iter, val_data_iter = load_cifar10_data_iter(batch_size=64, path=args.data_path)

    unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())

    for epoch in range(125):
        # anneal learning rate
        if epoch in (75, 100):
            updater.learning_rate = updater.learning_rate * 0.1
            print 'epoch %d learning rate annealed to %f' % (epoch, updater.learning_rate)

        t0 = time.time()
        forward_time, backward_time, updating_time = 0, 0, 0

        # training
        train_data_iter.reset() # data iterator must be reset every epoch
        for iteration, batch in enumerate(train_data_iter):

            data, labels = unpack_batch(batch)
            arg_dict['data'][:] = data
            arg_dict['softmax_label'][:] = labels

            t1 = time.time()
            executor.forward(is_train=True)
            # TODO loss
            forward_time += time.time() - t1

            t2 = time.time()
            executor.backward()
            backward_time += time.time() - t2

            t3 = time.time()
            for arg, grad in args_grad.items():
                momentem_update(arg_dict[arg], grad, cache_dict[arg], lr, momentem)
            updating_time += time.time() - t3

            if (iteration + 1) % 100 == 0:
                print 'epoch %d iteration %d loss %f' % (epoch, iteration + 1, loss[0])

        print 'epoch %d %f seconds consumed' % (epoch, time.time() - t0)
        print 'forward %f' % forward_time
        print 'backward %f' % backward_time
        print 'updating %f' % updating_time

        # validation
        val_data_iter.reset() # data iterator must be reset every epoch
        n_errors, n_samples = 0.0, 0.0
        for batch in val_data_iter:

            data, labels = unpack_batch(batch)
            arg_dict['data'][:] = data
            arg_dict['softmax_label'][:] = labels

            probs = executor.forward(is_train=False)
            preds = mx.nd.argmax(probs, axis=1)
            n_errors += mx.nd.sum(preds != labels)
            n_samples += len(data)

        print 'epoch %d validation error %f' % (epoch, n_errors / n_samples)
