from minpy import core
import minpy.numpy as np

def f(x):
    return x


if __name__ == '__main__':
    inp = np.random.random((3, 2))
    print(inp.asnumpy())
    g0 = core.grad(f)(inp)
    # All ones.
    print(g0.asnumpy())
    g_inj = np.random.random((3, 2))
    print(g_inj.asnumpy())
    injection = lambda *args, **kwargs: f(*args, **kwargs) * g_inj
    g1 = core.grad(injection)(inp)
    # Gradient will be as injected.
    print(g1.asnumpy())
