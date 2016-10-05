def test_clip():
    a = np.arange(10)
    np.clip(a, 1, 8)
    np.clip(a, 3, 6, out=a)
    a = np.arange(10)
    np.clip(a, [3,4,1,1,1,4,4,4,4,4], 8)

def test_concatenate():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    np.concatenate((a, b), axis=0)
    np.concatenate((a, b.T), axis=1)

    a = np.ma.arange(3)
    a[1] = np.ma.masked
    b = np.arange(2, 5)
    np.concatenate([a, b])
    np.ma.concatenate([a, b])

def test_floor():
    a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    np.floor(a)

def test_ceil():
    a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    np.ceil(a)

def test_full():
    np.full((2, 2), np.inf)
    np.full((2, 2), 10, dtype=np.int)

def test_flip():
    A = np.arange(8).reshape((2,2,2))
    flip(A, 0)
    flip(A, 1)
    A = np.random.randn(3,4,5)
    np.all(flip(A,2) == A[:,:,::-1,...])

def test_gamma():
    shape, scale = 2., 2. # mean and dispersion
    s = np.random.gamma(shape, scale, 1000)

def test_cos():
    np.cos(np.array([0, np.pi/2, np.pi]))
    # Example of providing the optional output parameter
    out2 = np.cos([0.1], out1)

def test_dot():
    np.dot(3, 4)
    a = [[1, 0], [0, 1]]
    b = [[4, 1], [2, 2]]
    np.dot(a, b)
    a = np.arange(3*4*5*6).reshape((3,4,5,6))
    b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    np.dot(a, b)[2,3,2,1,2,2]
    sum(a[2,3,2,:] * b[1,2,:,2])

def test_multiply():
    np.multiply(2.0, 4.0)
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)
    np.multiply(x1, x2)

def test_divide():
    np.divide(2.0, 4.0)
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)
    np.divide(x1, x2)
    np.divide(2, 4)
    np.divide(2, 4.)

def test_true_divide():
    x = np.arange(5)
    np.true_divide(x, 4)
    from __future__ import division
    x/4
    x//4

def test_ones():
    np.ones(5)
    np.ones((5,))
    np.ones((2, 1))
    s = (2,2)
    np.ones(s)

def test_zeros():
    np.zeros(5)
    np.zeros((5,))
    np.zeros((2, 1))
    s = (2,2)
    np.zeros(s)

def test_empty():
    np.empty([2, 2])

def test_uniform():
    s = np.random.uniform(-1,0,1000)
    np.all(s >= -1)
    np.all(s < 0)

def test_transpose():
    x = np.arange(4).reshape((2,2))
    x.transpose(x)
    x = np.ones((1, 2, 3))
    np.transpose(x, (1, 0, 2)).shape
