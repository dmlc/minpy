import numpy as np

""" Generates several clusters of Gaussian points """
def gaussian_cluster_generator(num_samples=10000, num_features=500, num_classes=5):
    mu = np.random.rand(num_classes, num_features)
    sigma = np.ones((num_classes, num_features)) * 0.1
    num_cls_samples = num_samples / num_classes
    x = np.zeros((num_samples, num_features))
    y = np.zeros((num_samples, num_classes))
    for i in range(num_classes):
        cls_samples = np.random.normal(mu[i,:], sigma[i,:], (num_cls_samples, num_features))
        x[i*num_cls_samples:(i+1)*num_cls_samples] = cls_samples
        y[i*num_cls_samples:(i+1)*num_cls_samples,i] = 1
    return x, y
