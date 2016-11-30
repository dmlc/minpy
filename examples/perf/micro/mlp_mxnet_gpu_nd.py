"""Simple multi-layer perception neural network on MNIST."""
import argparse
import os.path
import struct
import random
import time
import mxnet as mx

num_cold = 5

def init(args):
    layers = [784] + [args.hidden_size] * args.num_hidden + [10]
    biases = [mx.nd.zeros((1, x), ctx=mx.gpu(0)) for x in layers[1:]]
    weights = [
        mx.nd.zeros((x, y), ctx=mx.gpu(0)) for x, y in zip(layers[:-1], layers[1:])
    ]
    return weights, biases

def forward(inputs, weights, biases):
    activation = inputs
    activations = [inputs]
    for b, w in zip(biases[:-1], weights[:-1]):
        z = mx.nd.dot(activation, w) + b
        activation = mx.nd.maximum(z, 0)
        activations.append(activation)
    activation = mx.nd.dot(activation, weights[-1]) + biases[-1]
    activations.append(activation)
    return activations


def backward(g, activations, weights, biases):
    weight_deltas = []
    bias_deltas = []
    for weight, activation in reversed(list(zip(weights, activations))):
        bias_deltas.append(mx.nd.sum(g, axis=0, keepdims=True))
        weight_deltas.append(mx.nd.dot(activation.T, g))
        g = mx.nd.dot(g, weight.T)
        g = g * (activation > 1e-4)
    return list(reversed(bias_deltas)), list(reversed(weight_deltas))


def softmax(activation, one_hot):
    n = activation.shape[0]
    probs = activation - mx.nd.max(activation, axis=1, keepdims=True)
    loss = -mx.nd.sum(probs * one_hot - mx.nd.log(
        mx.nd.sum(mx.nd.exp(probs), axis=1, keepdims=True))) / n
    return loss


def accuracy(activation, label):
    return mx.nd.sum(mx.nd.argmax(
        activation, axis=1) == label) / float(activation.shape[0])


def softmax_loss_gradient(activation, one_hot):
    if False:
        n = activation.shape[0]
        m = mx.nd.amax(activation, axis=1, keepdims=True)
        probs = activation - m
        exp = mx.nd.exp(probs)
        loss = -mx.nd.sum(probs * one_hot - mx.nd.log(
            mx.nd.sum(exp, axis=1, keepdims=True))) / n
        g = -1 / n * (mx.nd.ones_like(activation) * one_hot - mx.nd.broadcast_to(
            1 / mx.nd.sum(exp, axis=1, keepdims=True), activation.shape) * exp)
        g = g * (1 - (mx.nd.broadcast_to(m, activation.shape) == activation))
        return g
    else:
        probs = activation - mx.nd.max(activation, axis=1, keepdims=True)
        e = mx.nd.exp(probs)
        p = e / mx.nd.sum(e, axis=1, keepdims=True)
        q = p - one_hot
        return q


def main(args):
    weights, biases = init(args)
    img = mx.nd.zeros((args.batch_size, 784), ctx=mx.gpu(0))
    label = mx.nd.zeros((args.batch_size,), ctx=mx.gpu(0))
    for l in range(args.num_loops):
        if l == num_cold:
            start = time.time()
        f = forward(img, weights, biases)
        activation = f[-1]
        if args.only_forward:
            activation.wait_to_read()
            continue
        num_samples = activation.shape[0]
        label_one_hot = mx.nd.zeros(activation.shape, ctx=mx.gpu(0))
        mx.nd.onehot_encode(label, label_one_hot)
        g = softmax_loss_gradient(activation, label_one_hot)
        bias_deltas, weight_deltas = backward(g, f, weights, biases)
        for b_delta in bias_deltas:
            b_delta.wait_to_read()
        for w_delta in weight_deltas:
            w_delta.wait_to_read()
    dur = time.time() - start
    print('Per Loop Time: %.6f' % (dur / (args.num_loops - num_cold)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-forward', default=False, action='store_true')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--hidden-size', default=256, type=int)
    parser.add_argument('--num-hidden', default=1, type=int)
    parser.add_argument('--num-loops', default=20, type=int)
    #import profile
    #profile.run('main(parser.parse_args())')
    main(parser.parse_args())
