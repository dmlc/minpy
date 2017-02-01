Select Policy for Operations
============================

MinPy integrates MXNet NDArray and NumPy into a seamless system. For a single operation, it may have MXNet
implementation, pure NumPy CPU implementation, or both of them. MinPy utilizes a policy system to determine which
implementation will be applied. MinPy currently has three build-in policies:

1. ``prefer_mxnet`` [**Default**]: Prefer MXNet. Use NumPy as a transparent fallback.
2. ``only_numpy``: Only use NumPy.
3. ``only_mxnet``: Only use MXNet.

The policy is global. To change the policy, use ``minpy.set_global_policy``. For example:

::

    import minpy.numpy as np
    import minpy
    minpy.set_global_policy('only_numpy')

It is worth mentioning that ``minpy.set_global_policy`` only accepts strings of policy names.

``@minpy.wrap_policy``: Wrap a Function under Specific Policy
-------------------------------------------------------------
``@minpy.wrap_policy`` is a wrapper that wraps a function under specific policy. It only accpets policy objects.
For example

::

    import minpy.numpy as np
    import minpy
    import minpy.dispatch.policy as policy
    minpy.set_global_policy('prefer_mxnet')


    @minpy.wrap_policy(policy.OnlyNumPyPolicy())
    def foo(a, b):
        return np.log(a + b)

    a = np.ones((2, 2))
    b = np.zeros((2, 2))

    # a + b runs under PreferMXNetPolicy
    c = a + b

    # foo runs under OnlyNumPyPolicy.
    c = foo(np.ones((2, 2)), np.zeros((2, 2)))

    # a + b runs under PreferMXNetPolicy again
    c = a + b
