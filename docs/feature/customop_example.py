@customop('numpy')
def my_softmax(x, y):
    probs = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
    probs /= numpy.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -numpy.sum(numpy.log(probs[numpy.arange(N), y])) / N
    return loss


def my_softmax_grad(ans, x, y):
    def grad(g):
        N = x.shape[0]
        probs = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
        probs /= numpy.sum(probs, axis=1, keepdims=True)
        probs[numpy.arange(N), y] -= 1
        probs /= N
        return probs

    return grad

my_softmax.def_grad(my_softmax_grad)
