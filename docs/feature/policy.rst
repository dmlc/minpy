Select Policy for Operations
============================

MinPy integrates MXNet NDArray and NumPy into a seamless system. For a single operation, it may have MXNet
implementation, pure NumPy CPU implementation, or both of them. MinPy utilizes a policy system to determine which
implementation will be applied. MinPy has three build-in policies in ``minpy.dispatch.policy``:

1. ``PreferMXNetPolicy()`` (**Default**): Prefer MXNet. Use NumPy as a transparent fallback.
2. ``OnlyNumPyPolicy()``: Only use NumPy.
3. ``OnlyMXNetPolicy()``: Only use MXNet.

To change policy, use ``minpy.numpy.set_policy`` method. For example:
::

    import minpy.numpy as np
    import minpy.dispatch.policy as policy
    np.set_policy(policy.OnlyNumPyPolicy())

It is worth mentioning that ``np.set_policy`` only accepts instances of policy classes.

The policy is active in the lifetime of the current imported MinPy module, which is usually the scope of the current
file.