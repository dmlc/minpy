def gru_step(x, prev_h, Wx, Wh, b, Wxh, Whh, bh):
    """
    Forward pass for a single timestep of an GRU.

    The input data has dimentsion D, the hidden state has dimension H, and we
    use a minibatch size of N.

    Parameters
    ----------
    x : Input data, of shape (N, D)
    prev_h : Previous hidden state, of shape (N, H)
    prev_c : Previous hidden state, of shape (N, H)
    Wx : Input-to-hidden weights for r and z gates, of shape (D, 2H)
    Wh : Hidden-to-hidden weights for r and z gates, of shape (H, 2H)
    b : Biases for r an z gates, of shape (2H,)
    Wxh : Input-to-hidden weights for h', of shape (D, H)
    Whh : Hidden-to-hidden weights for h', of shape (H, H)
    bh : Biases, of shape (H,)

    Returns
    -------
    next_h : Next hidden state, of shape (N, H)

    Notes
    -----
    Implementation follows
    http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    """
    N, H = prev_h.shape
    a = sigmoid(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
    r = a[:, 0:H]
    z = a[:, H:2 * H]
    h_m = np.tanh(np.dot(x, Wxh) + np.dot(r * prev_h, Whh) + bh)
    next_h = z * prev_h + (1 - z) * h_m
    return next_h