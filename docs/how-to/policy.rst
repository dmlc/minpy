Select Policy of Operation
==========================

MinPy integrates MXNet NDArray and NumPy into a seamless system. For a single operation, it may have MXNet
implementation on GPU, pure NumPy CPU implementation, or both. MinPy utilizes policy to determine which
implementation will be applied. MinPy has three build-in policies in ``minpy.dispatch.policy``:

1. ``PreferMXNetPolicy`` (**Default**): Prefer MXNet implementation. Use NumPy as a fallback.
2. ``OnlyNumPyPolicy``: Only use NumPy implementation.
3. ``OnlyMXNetPolicy``: Only use MXNet implementation.

To change policy, use ``np.set_policy``. For example:
::

    import minpy.numpy as np
    import minpy.dispatch.policy as policy
    np.set_policy(policy.OnlyNumPyPolicy)

The policy will be active in its scope.
