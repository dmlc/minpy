"""Simple multi-layer perception neural network on MNIST."""
import argparse
import os.path
import struct
import random
import time
import numpy as np

num_cold = 5

def init(args):
    layers = [784] + [args.hidden_size] * args.num_hidden + [10]
    biases = [np.random.normal(scale=0.001, size=(1, x)) for x in layers[1:]]
    weights = [
        np.random.normal(
            scale=0.001, size=(x, y)) for x, y in zip(layers[:-1], layers[1:])
    ]
    return weights, biases

def forward(inputs, weights, biases):
    activation = inputs
    activations = [inputs]
    for b, w in zip(biases[:-1], weights[:-1]):
        z = np.dot(activation, w) + b
        activation = np.maximum(z, 0)
        activations.append(activation)
    activation = np.dot(activation, weights[-1]) + biases[-1]
    activations.append(activation)
    return activations


def backward(g, activations, weights, biases):
    weight_deltas = []
    bias_deltas = []
    for weight, activation in reversed(list(zip(weights, activations))):
        bias_deltas.append(np.sum(g, axis=0, keepdims=True))
        weight_deltas.append(activation.T.dot(g))
        g = np.dot(g, weight.T)
        g = g * (activation > 1e-4)
    return list(reversed(bias_deltas)), list(reversed(weight_deltas))


def softmax(activation, one_hot):
    n = activation.shape[0]
    probs = activation - np.amax(activation, axis=1, keepdims=True)
    loss = -np.sum(probs * one_hot - np.log(
        np.sum(np.exp(probs), axis=1, keepdims=True))) / n
    return loss


def accuracy(activation, label):
    return np.sum(np.argmax(
        activation, axis=1) == label) / float(activation.shape[0])


def softmax_loss_gradient(activation, one_hot):
    if False:
        n = activation.shape[0]
        m = np.amax(activation, axis=1, keepdims=True)
        probs = activation - m
        exp = np.exp(probs)
        loss = -np.sum(probs * one_hot - np.log(
            np.sum(exp, axis=1, keepdims=True))) / n
        g = -1 / n * (np.ones_like(activation) * one_hot - np.broadcast_to(
            1 / np.sum(exp, axis=1, keepdims=True), activation.shape) * exp)
        g = g * (1 - (np.broadcast_to(m, activation.shape) == activation))
        return g
    else:
        probs = activation - np.amax(activation, axis=1, keepdims=True)
        e = np.exp(probs)
        p = e / np.sum(e, axis=1, keepdims=True)
        q = p - one_hot
        return q


def main(args):
    weights, biases = init(args)
    img = np.zeros((args.batch_size, 784))
    label = np.zeros((args.batch_size,), dtype=np.int)
    for l in range(args.num_loops):
        if l == num_cold:
            start = time.time()
        f = forward(img, weights, biases)
        activation = f[-1]
        if args.only_forward:
            continue
        num_samples = activation.shape[0]
        label_one_hot = np.zeros_like(activation)
        label_one_hot[np.arange(num_samples), label] = 1
        g = softmax_loss_gradient(activation, label_one_hot)
        bias_deltas, weight_deltas = backward(g, f, weights, biases)
        #for b, b_delta in zip(biases, bias_deltas):
            #b -= 0.01 * b_delta / num_samples
        #for w, w_delta in zip(weights, weight_deltas):
            #w -= 0.01 * w_delta / num_samples
    dur = time.time() - start
    print('Per Loop Time: %.6f' % (dur / (args.num_loops - num_cold)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-forward', default=False, action='store_true')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--hidden-size', default=256, type=int)
    parser.add_argument('--num-hidden', default=1, type=int)
    parser.add_argument('--num-loops', default=20, type=int)
    main(parser.parse_args())
