from __future__ import print_function
import minpy.numpy as np
from minpy.core import grad


def func(x):
	y = x + 1
	return x

x = 1
gradient = grad(func)
print('Gradient should be 1, and it is actually', gradient(x))
