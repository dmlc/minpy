import minpy.numpy as np
import minpy.numpy.random as random
from minpy.core import grad_and_loss
from examples.utils.data_utils import gaussian_cluster_generator as make_data
from minpy.context import set_context, gpu

# Please uncomment following if you have GPU-enabled MXNet installed.
# This single line of code will run MXNet operations on GPU 0.
# set_context(gpu(0)) # set the global context as gpu(0)

# Predict the class using multinomial logistic regression (softmax regression).
def predict(w, x):
    a = np.exp(np.dot(x, w))
    a_sum = np.sum(a, axis=1, keepdims=True)
    prob = a / a_sum
    return prob

def train_loss(w, x):
    prob = predict(w, x)
    loss = -np.sum(label * np.log(prob)) / num_samples
    return loss

"""Use Minpy's auto-grad to derive a gradient function off loss"""
grad_function = grad_and_loss(train_loss)

# Using gradient descent to fit the correct classes.
def train(w, x, loops):
    for i in range(loops):
        dw, loss = grad_function(w, x)
        if i % 10 == 0:
            print('Iter {}, training loss {}'.format(i, loss))
        # gradient descent
        w -= 0.1 * dw

# Initialize training data.
num_samples = 10000
num_features = 500
num_classes = 5
data, label = make_data(num_samples, num_features, num_classes)

# Initialize training weight and train
weight = random.randn(num_features, num_classes)
train(weight, data, 100)
