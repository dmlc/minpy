import numpy as np
import numpy.random as random

""" Generates several clusters of Gaussian points """
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

# Predict the class using logistic regression.
def predict(w, x):
    a = np.exp(np.dot(x, w))
    a_sum = np.sum(a, axis=1, keepdims=True)
    prob = a / a_sum
    return prob

# Using gradient descent to fit the correct classes.
def train(w, x, loops):
    for i in range(loops):
        prob = predict(w, x)
        loss = -np.sum(label * np.log())
        if i % 10 == 0:
            print('Iter {}, training loss {}'.format(i, loss))
        # gradient descent
        dy = prob - label
        dw = np.dot(data.T, dy) / num_samples
        # update parameters; fixed learning rate of 0.1
        w -= 0.1 * dw

# Initialize training data.
num_samples = 10000
num_features = 500
num_classes = 5
data, label = make_data(num_samples, num_features, num_classes)

# Initialize training weight and trian
weight = random.randn(num_features, num_classes)
train(weight, data, 100)
