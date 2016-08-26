Select Context for MXNet
========================

MinPy as a system fully integrates MXNet, enjoys MXNet's flexibility to run operations on CPU and different GPUs. The
``Context`` in MinPy determines where MXNet operations run. MinPy has two built-in ``Context`` in ``minpy.context``:

1. ``minpy.context.cpu()`` [**Default**]: runs on CPU. No ``device_id`` needed for CPU context.

2. ``minpy.context.gpu(device_id)``: runs on GPU specified by ``device_id``. Usually ``gpu(0)`` is the first GPU in the system. Note that GPU context is only available with MXNet complied with GPU support.


There are two functions to set context:

1. use ``minpy.context.set_context`` to set global context. we encourage you to use it at the header of program. For example:
::
    from minpy.context import set_context, cpu, gpu
    set_context(gpu(0))  # set the global context as gpu(0)

It is worth mentioning that ``minpy.context.set_context`` only accepts instances of context classes.

The context is active in the lifetime of the current imported MinPy module, which is usually the scope of the current file.

2. use ``with`` statement to set local context. For example:
::
    with gpu(0):
        x_gpu0 = random.rand(32, 64) - 0.5
        y_gpu0 = random.rand(64, 32) - 0.5
        z_gpu0 = np.dot(x_gpu0, y_gpu0)
    with gpu(1):
        x_gpu1 = random.rand(32, 64) - 0.5
        y_gpu1 = random.rand(64, 32) - 0.5
        z_gpu1 = np.dot(x_gpu1, x_gpu1)

The code snippet will run on ``gpu0`` or ``gpu1`` decided by the device information in the with statement. With this feature, you can achive distributing computation on multi-device.

