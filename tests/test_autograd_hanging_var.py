import minpy.numpy as np
from minpy.core import grad


def func(x):
	y = x + 1
	return x

x = 1
gradient = grad(func)
print gradient(x)
