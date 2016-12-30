import sys

from minpy.core import grad
import minpy.numpy as np
import minpy.numpy.random as random
import minpy.dispatch.policy as policy

#np.set_policy(policy.OnlyNumPyPolicy())

def test_logistic():

    def sigmoid(x):
        return 0.5 * (np.tanh(x / 2) + 1)
    
    
    def predict(weights, inputs):
        return sigmoid(np.dot(inputs, weights))
    
    
    def training_loss(weights, inputs):
        preds = predict(weights, inputs)
        label_probabilities = preds * targets + (1 - preds) * (1 - targets)
        l = -np.sum(np.log(label_probabilities))
        return l
    
    
    def training_accuracy(weights, inputs):
        preds = predict(weights, inputs)
        error = np.count_nonzero(
            np.argmax(
                preds, axis=1) - np.argmax(
                    targets, axis=1))
        return (256 - error) * 100 / 256.0

    xshape = (256, 500)
    wshape = (500, 250)
    tshape = (256, 250)
    inputs = random.rand(*xshape) - 0.5
    targets = np.zeros(tshape)
    truth = random.randint(0, 250, 256)
    targets[np.arange(256), truth] = 1
    weights = random.rand(*wshape) - 0.5
    
    training_gradient_fun = grad(training_loss)
    
    for i in range(50):
        acc = training_accuracy(weights, inputs)
        print('Trained accuracy #{}: {}%'.format(i, acc))
        gr = training_gradient_fun(weights, inputs)
        weights -= gr * 0.01

    return acc > 99.0

if __name__ == "__main__":
    sys.exit(not test_logistic())
