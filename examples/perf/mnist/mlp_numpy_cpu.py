"""Simple multi-layer perception neural network on MNIST."""
import argparse
import os.path
import struct
import random
import time
import numpy as np

num_epochs = 10
mini_batch_size = 256

layers = [784, 256, 10]
biases = [np.random.normal(scale=0.001, size=(1, x)) for x in layers[1:]]
weights = [
    np.random.normal(
        scale=0.001, size=(x, y)) for x, y in zip(layers[:-1], layers[1:])
]
alpha = 1e-3


def forward(inputs):
    activation = inputs
    activations = [inputs]
    for b, w in zip(biases[:-1], weights[:-1]):
        z = np.dot(activation, w) + b
        activation = np.maximum(z, 0)
        activations.append(activation)
    activation = np.dot(activation, weights[-1]) + biases[-1]
    activations.append(activation)
    return activations


def backward(g, activations):
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
    img_fname = os.path.join(args.data_dir, 'train-images-idx3-ubyte')
    label_fname = os.path.join(args.data_dir, 'train-labels-idx1-ubyte')
    with open(label_fname, 'rb') as f:
        magic_nr, size = struct.unpack('>II', f.read(8))
        assert magic_nr == 2049
        assert size == 60000
        label = np.fromfile(f, dtype=np.int8)
    with open(img_fname, 'rb') as f:
        magic_nr, size, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic_nr == 2051
        assert size == 60000
        assert rows == cols == 28
        img = np.fromfile(f, dtype=np.uint8).reshape(size, rows * cols)
    for epoch in range(num_epochs):
        print('Epoch {}.'.format(epoch))
        indices = list(range(len(label)))
        random.shuffle(indices)
        img_mini_batches = [
            img[indices[k:k + mini_batch_size]]
            for k in range(0, len(img), mini_batch_size)
        ]
        label_mini_batches = [
            label[indices[k:k + mini_batch_size]]
            for k in range(0, len(img), mini_batch_size)
        ]
        start = time.time()
        for img_mini_batch, label_mini_batch in zip(img_mini_batches,
                                                    label_mini_batches):
            f = forward(img_mini_batch)
            activation = f[-1]
            num_samples = activation.shape[0]
            label_one_hot = np.zeros_like(activation)
            label_one_hot[np.arange(num_samples), label_mini_batch] = 1
            g = softmax_loss_gradient(activation, label_one_hot)
            bias_deltas, weight_deltas = backward(g, f)
            for b, b_delta in zip(biases, bias_deltas):
                b -= alpha * b_delta / num_samples
            for w, w_delta in zip(weights, weight_deltas):
                w -= alpha * w_delta / num_samples
            # loss = softmax(activation, label_one_hot)
            # print(loss)
        print('Time: {}.'.format(time.time() - start))
        print(accuracy(activation, label_mini_batch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    main(parser.parse_args())
