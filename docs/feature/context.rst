Select Context for MXNet
========================

MinPy as a system fully integrates MXNet, enjoys MXNet's flexibility to run operations on CPU and different GPUs. The
``Context`` in MinPy determines where MXNet operations run. MinPy has two built-in ``Context`` in ``minpy.context``:

1. ``gpu(device_id)``: runs on GPU specified by ``device_id``. Usually ``gpu(0)`` is the first GPU in the system.
Note that GPU context is only available with MXNet complied with GPU support.
2. ``cpu()`` (**Default**): runs on CPU. No ``device_id`` needed for CPU context.

To set context, use function ``minpy.context.set_context``. For example:
::

    from minpy.context import set_context, gpu
    set_context(gpu(0))  # set the global context as gpu(0)

It is worth mentioning that ``minpy.context.set_context`` only accepts instances of context classes.

The context is active in the lifetime of the current imported MinPy module, which is usually the scope of the current
file.