Select Policy for Operations
============================

MinPy integrates MXNet NDArray and NumPy into a seamless system. For a single operation, it may have MXNet
implementation, pure NumPy CPU implementation, or both of them. MinPy utilizes a policy system to determine which
implementation will be applied. MinPy has three build-in policies in ``minpy.dispatch.policy``
(also aliased in ``minpy`` root):

1. ``PreferMXNetPolicy()`` [**Default**]: Prefer MXNet. Use NumPy as a transparent fallback.
2. ``OnlyNumPyPolicy()``: Only use NumPy.
3. ``OnlyMXNetPolicy()``: Only use MXNet.

The policy is set under module level. Each mocking module (a.k.a. the module
mocking the behavior of NumPy's corresponding package), like ``minpy.numpy``
and ``minpy.numpy.random`` has its own policy. To change the policy, use
``set_policy`` method in the module. For example, for ``minpy.numpy``, use
``minpy.numpy.set_policy`` method:

::

    import minpy.numpy as np
    from minpy import OnlyNumPyPolicy
    np.set_policy(OnlyNumPyPolicy())

To make life easier, MinPy can also change the policy of all MinPy mocking
modules at the same time, including the modules already imported and modules
imported in the future. Simply add the following two lines before computation.
Note that ``minpy.set_global_policy`` changes **all** MinPy mocking modules
and set a new default.

::

    import minpy.numpy as np
    import minpy.numpy.random as random
    import minpy
    minpy.set_global_policy(minpy.OnlyNumPyPolicy())
    # np and random is now in OnlyNumPyPolicy policy

It is worth mentioning that ``np.set_policy`` and ``minpy.set_global_policy`` only accept instances of policy classes.

``@minpy.wrap_policy``: Wrap a Function under Specific Policy
-------------------------------------------------------------
``@minpy.wrap_policy`` is a wrapper that wraps a function under specific policy. For example

::

    import minpy.numpy as np
    import minpy
    minpy.set_global_policy(minpy.PreferMXNetPolicy())


    @minpy.wrap_policy(minpy.OnlyNumPyPolicy())
    def foo(a, b)
        return np.log(a + b)

    a = np.ones((2, 2))
    b = np.zeros((2, 2))

    # a + b runs under PreferMXNetPolicy
    c = a + b

    # foo runs under OnlyNumPyPolicy.
    c = foo(np.ones((2, 2)), np.zeros((2, 2)))

    # a + b runs under PreferMXNetPolicy again
    c = a + b
