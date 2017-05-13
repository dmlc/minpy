import numpy as np

def foo(x, y):
    return np.dot(x, y)

def bar(*args, **kwargs):
    return foo(*args, **kwargs)

x = np.zeros((256, 512))
y = np.zeros((512, 512))

def example3():
  for i in range(0,1000):
    bar(x, y)

import profile
profile.run("example3()")
