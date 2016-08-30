Select Policy for Operations
============================

MinPy integrates MXNet NDArray and NumPy into a seamless system. For a single operation, it may have MXNet
implementation, pure NumPy CPU implementation, or both of them. MinPy utilizes a policy system to determine which
implementation will be applied. MinPy has three build-in policies in ``minpy.dispatch.policy``
(also aliased in ``minpy`` root):

1. ``PreferMXNetPolicy()`` [**Default**]: Prefer MXNet. Use NumPy as a transparent fallback.
2. ``OnlyNumPyPolicy()``: Only use NumPy.
3. ``OnlyMXNetPolicy()``: Only use MXNet.

The policy is set under module level. Each mocking module (a.k.a. the module mocking the behavior of NumPy's corresponding
package), like ``minpy.numpy`` and ``minpy.numpy.random`` has its own policy. To change the policy, use ``set_policy`` method
in the module. For example, for ``minpy.numpy``, use ``minpy.numpy.set_policy`` method:

::

    import minpy.numpy as np
    from minpy import OnlyNumPyPolicy
    np.set_policy(OnlyNumPyPolicy())

To make life easier, MinPy can also change the policies of all MinPy mocking modules at the same time. Simple add the following two lines before computation.

::

    import minpy
    minpy.set_global_policy(minpy.OnlyNumPyPolicy())

It is worth mentioning that ``np.set_policy`` and ``minpy.set_global_policy`` only accept instances of policy classes.
