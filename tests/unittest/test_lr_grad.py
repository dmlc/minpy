import sys
import numpy

import minpy.numpy as np
from minpy.core import grad
from minpy.utils import gradient_checker
import minpy.dispatch.policy as policy
#np.set_policy(policy.OnlyNumPyPolicy())

rng = numpy.random.RandomState(42)

def test_lr_grad():
    inputs = rng.rand(32, 64) * 0.1
    targets = np.zeros((32, 10))
    truth = rng.randint(0, 10, 32)
    targets[np.arange(32), truth] = 1
    
    def sigmoid(x):
        return 0.5 * (np.tanh(x / 2) + 1)
    
    def training_loss(weights):
        preds = sigmoid(np.dot(inputs, weights))
        label_probabilities = preds * targets + (1 - preds) * (1 - targets)
        l = -np.sum(np.log(label_probabilities))
        return l
    
    weights = rng.rand(64, 10) * 0.01

    return gradient_checker.quick_grad_check(training_loss, weights, rs=rng)

if __name__ == "__main__":
    sys.exit(not test_lr_grad())
