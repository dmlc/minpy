Operation Policy Selection
==========================

MinPy integrates MXNet NDArray and NumPy into a seamless system. For a single operation, it may have MXNet
implementation on GPU, or pure NumPy CPU implementation, or both. NumPy has three policies in ``minpy.dispatch.policy``:

1. ``PreferMXNetPolicy`` (Default): Prefer MXNet implementation. Use NumPy as a fallback.
2. ``OnlyNumPyPolicy``: Only use NumPy implementation.
3. ``OnlyMXNetPolicy``: Only use MXNet implementation.

To change policy, use ``np.set_policy``. For example:

    import minpy.numpy as np
    import minpy.dispatch.policy as policy
    np.set_policy(policy.OnlyNumPyPolicy)

The policy will be active in current scope.