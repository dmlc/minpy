import math
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def make_data(num_samples=10000, num_features=500, num_classes=5):
    mu = random.rand(num_classes, num_features)
    sigma = np.ones((num_classes, num_features)) * 0.1
    num_cls_samples = num_samples / num_classes
    x = np.zeros((num_samples, num_features))
    y = np.zeros((num_samples, num_classes))
    for i in range(num_classes):
        cls_samples = random.normal(mu[i,:], sigma[i,:], (num_cls_samples, num_features))
        x[i*num_cls_samples:(i+1)*num_cls_samples] = cls_samples
        y[i*num_cls_samples:(i+1)*num_cls_samples,i] = 1
    return x, y

def make_weight(num_features=500, num_classes=5):
    w = np.random.randn(num_features, num_classes)
    return w

def get_data():
    x = np.load('x.npy')
    y = np.load('y.npy')
    return x, y

def get_weight():
    return np.load('w.npy')

def plot_data(x, y, num_classes=5):
    t = np.argmax(y, axis=1)
    colors = ['r', 'b', 'g', 'c', 'y']
    for i in range(num_classes):
        cls_x = x[t == i]
        plt.scatter(cls_x[:,0], cls_x[:,1], color=colors[i], s=1)
    plt.show()
