import numpy as np

def foo(x, y):
    return np.dot(x, y)

x = np.zeros((256, 512))
y = np.zeros((512, 512))

def example():
  for i in range(0,1000):
    foo(x, y)

import profile
profile.run("example()")
