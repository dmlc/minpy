# coding: utf-8
# A bit of setup, just ignore this cell.
from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import numpy as np
import numpy.random as random
import time
from util import get_data, plot_data

# Initialize training data.
data, label = get_data()
num_samples = data.shape[0]
num_features = data.shape[1]
num_classes = label.shape[1]

print('Shapes: data {}, label {}'.format(data.shape, label.shape))
print('#samples: {}'.format(num_samples))
print('#features: {}'.format(num_features))
print('#classes: {}'.format(num_classes))


print('Data matrix:')
print(data)

print('Label matrix:')
print(label)

# Predict the class using logistic regression.
def predict(w, x):
    return np.dot(x, w)

# Initialize training weight.
weight = random.randn(num_features, num_classes)

# Using gradient descent to fit the correct classes.
def train(w, x, loops):
    for i in range(loops):
        prob = predict(w, x)

# Now training it for 100 iterations.
start_time = time.time()
train(weight, data, 100)
print('Training time: {}s'.format(time.time() - start_time))

# ## Multinomial Logistic Regression using MinPy (MXNet NumPy)
# 
# ### Utilize GPU computation with little (or no) NumPy syntax change
# 
# You could see even with such a tiny example, 100 iterations take around 9 seconds. In real world, there are billions of samples and much more features and classes. How to efficiently train such a model? One solution is to use GPU. Our tool, MinPy allows you to use GPU to speed up the algorithm and in the meantime, keep the neat NumPy syntax you just went through.

# In[10]:

# All you need to do is replace the NumPy namespace with MinPy's.
import minpy.numpy as np
import minpy.numpy.random as random

import minpy.dispatch.policy as ply
np.set_policy(ply.OnlyNumPyPolicy())

# In[11]:

# Initialize weight matrix (again).
weight = random.randn(num_features, num_classes)

# Now call the same training function.
# Since the namespace is redefined, it will automatically run on GPU.
start_time = time.time()
train(weight, data, 100)
 # You should observe a significant speed up (around 3x) to the previous training time.
print('Training time: {}s'.format(time.time() - start_time))


# ### Automatic gradient calculation
# 
# Compute gradient is tedious especially for complex neural networks you will encounter in the following tutorials. Minpy is able to compute the gradient automatically given arbitrary loss function. Please implement the following loss function (or paste it from your previous codes). The `grad_and_loss` function takes your defined `train_loss` function and then returns a function which will calculate both the loss value and the gradient of weight. The gradient could then be directly used to update the model weight.

# **Quiz:** Try modify the loss function by adding an L2-regularization. The new loss function is as follows:
# $$J'(w)=J(w)+\sum_{i}w_i^2.$$

# In[12]:

'''
from minpy.core import grad_and_loss

# Initialize weight matrix (again).
weight = random.randn(num_features, num_classes)

# Using gradient descent to fit the correct classes.
def train_loss(w, x):
    #===========================================================#
    #                    Your code starts here                  #
    #===========================================================#
    prob = predict(w, x)
    loss = -np.sum(label * np.log(prob)) / num_samples
    #===========================================================#
    #                    Your code ends here                    #
    #===========================================================#
    return loss

# Calculate gradient function automatically.
grad_function = grad_and_loss(train_loss)

# Now training it for 100 iterations.
start_time = time.time()
for i in range(100):
    dw, loss = grad_function(weight, data)
    if i % 10 == 0:
        print('Iter {}, training loss {}'.format(i, loss))
    weight -= 0.1 * dw
print('Training time: {}s'.format(time.time() - start_time))
'''
