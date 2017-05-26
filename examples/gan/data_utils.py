import numpy as nnp
import cv2
from sklearn.datasets import fetch_mldata
import pickle

def get_mnist():
    mnist = fetch_mldata('MNIST original')
    nnp.random.seed(1234) # set seed for deterministic ordering
    p = nnp.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000, 28, 28))
    X = nnp.asarray([cv2.resize(x, (64,64)) for x in X])
    X = X.astype(nnp.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 64, 64))
    X = nnp.tile(X, (1, 3, 1, 1))
    X_train = X[:60000]
    X_test = X[60000:]
    return X_train, X_test

if __name__ == '__main__':
    X_train, X_test = get_mnist()
