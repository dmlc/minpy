import numpy as np

def foo(x, y):
    return np.dot(x, y)
    #return x + y

class Test:
    def __init__(self, f):
        self.f = f
    def __call__(self, *args, **kwargs):
        #args1 = tuple(a for a in args)
        #kwargs1 = {k: kwargs[k] for k in kwargs}
        self.f(*args, **kwargs)

a = Test(foo)
x = np.zeros((256, 512))
y = np.zeros((512, 512))

def example2():
  for i in range(0,100):
    a(x, y)

import profile
profile.run("example2()")
