"""Test customized operator."""
from __future__ import print_function
import minpy.numpy as np
from minpy.array import Number
from minpy.primitive import customop
from minpy.core import grad


def rel_error(x, y):
    """Returns relative error"""
    if isinstance(x, (int, float, Number)):
        x = float(x)
        y = float(y)
        return abs(x - y) / max(1e-8, abs(x) + abs(y))
    else:
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


@customop('numpy')
def softmax(x, y):
    import numpy as np
    y = y.astype(int)
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    return loss


def softmax_grad(ans, x, label):
    def grad(g):
        import numpy as np
        y = label.astype(int)
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N = x.shape[0]
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return dx

    return grad

softmax.def_grad(softmax_grad)


def test_customop():
    x = np.array([[0.24854138, 1.94385293, 2.33848549, 2.75407309, 1.66905118],
                  [0.26498274, 1.60618255, 1.25387436, 2.9215846, 1.26427169],
                  [0.87108803, 1.45227827, 2.17339809, 0.50049702, 0.5883466],
                  [0.66406034, 0.48855862, 1.53960508, 0.66568797, 2.23948055],
                  [2.72220612, 1.82959485, 1.51552618, 1.54757016, 1.64023012],
                  [1.69430802, 2.21234513, 0.44159807, 1.94465274, 0.11623679],
                  [0.71774937, 1.99183721, 2.93154152, 0.23254174, 1.63623933],
                  [1.54450952, 2.32885258, 1.64220968, 1.66349828, 2.50975782],
                  [0.99172053, 2.60171951, 1.14377575, 0.28264201, 2.50368237],
                  [1.99669231, 2.16996937, 1.77290071, 1.34783694, 2.42391734]])

    label = np.array([4, 0, 0, 1, 0, 4, 0, 2, 1, 3])
    grad_func = grad(softmax)

    # Check forward
    assert rel_error(softmax(x, label), 2.16612911224) < 1e-12

    expected_out = np.array([[0.00323393, 0.01761957, 0.02614461, 0.0396159, -0.08661401],
                             [-0.09591437, 0.01562194, 0.01098321, 0.05821122, 0.011098],
                             [-0.08735776, 0.02260642, 0.04649542, 0.00872727, 0.00952864],
                             [0.00992692, -0.09167096, 0.02382641, 0.00994309, 0.04797453],
                             [-0.05756653, 0.01738011, 0.01269563, 0.01310903, 0.01438177],
                             [0.02244518, 0.03767938, 0.00641325, 0.02883012, -0.09536792],
                             [-0.09406418, 0.02122317, 0.05431486, 0.00365391, 0.01487223],
                             [0.01242947, 0.02723256, -0.08629487, 0.01400002, 0.03263282],
                             [0.00820025, -0.05897573, 0.00954693, 0.00403532, 0.03719322],
                             [0.01982428, 0.02357495, 0.01584915, -0.08963896, 0.03039058]])

    # Check backward
    assert rel_error(grad_func(x, label), expected_out) < 1e-7
    print('All passed!')


if __name__ == '__main__':
    test_customop()
