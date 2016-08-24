Difference Between Policy and Context
=====================================
Policy dispatches operations between NumPy implementation and MXNet implementation. Context determines whether MXNet
implementation runs on GPU or CPU. If the policy sets to ``OnlyNumPyPolicy()``, then MinPy will only use NumPy
implementation. In this case, context setting will affect MinPy behavior.

If you have GPU-enabled MXNet installed, add the following settings immediately after MinPy related imports:
::

    from minpy.context import set_context, gpu
    set_context(gpu(0))  # set the global context as gpu(0)

With these two lines, MinPy's MXNet implementation will run at GPU 0. Since the default policy is ``OnlyNumPyPolicy()``,
MinPy will run on GPU via MXNet for supported operations and transparently fallback to NumPy.