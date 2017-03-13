import minpy
import minpy.numpy as np
import minpy.numpy.random as random
from minpy.core import grad_and_loss
# from examples.utils.data_utils import gaussian_cluster_generator as make_data
# from minpy.context import set_context, gpu

# Please uncomment following if you have GPU-enabled MXNet installed.
# This single line of code will run MXNet operations on GPU 0.
# set_context(gpu(0)) # set the global context as gpu(0)

# Predict the class using multinomial logistic regression (softmax regression).


# Because MXNet's implementation does not support mu and sigma to be arrays
# (only scalar is supported right now), we need to change the policy for
# this function by `@minpy.wrap_policy`or `with minpy.OnlyNumPyPolicy(): ...`
""" Generates several clusters of Gaussian points """
def test_policy():
    @minpy.wrap_policy("only_numpy")
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
    data, label = gaussian_cluster_generator(num_samples, num_features, num_classes)
    
    # Initialize training weight and train
    weight = random.randn(num_features, num_classes)
    train(weight, data, 100)

if __name__ == "__main__":
    test_policy()
