Decorators for Functions
========================

``@convert_args``: Convert Function Input to MinPy types
--------------------------------------------------------

If your data is all generated under MinPy's namespace, this section is not for
you. However, most of our users use MinPy along with other data, and the
majority of data they are working on are in NumPy format. Although every operation in MinPy's
namespace and gradient function solver (``grad`` and ``grad_and_loss``) will
convert every NumPy input to MinPy input automatically for further processing, there are
some corner cases that the computation will still happen in NumPy's namespace
instead of MinPy's. For example

::

    import minpy.numpy as np
    def simple_add(a, b):
        return a + b

Now we declare two NumPy arrays:

::

    import numpy as npp
    a = npp.ones((10, 10))
    b = npp.zeros((10, 10))

If we pass ``a`` and ``b`` into function ``simple_add``. the add operation
will happen in NumPy's namespace. This is not the expected behavior.
Thus we need to provide a tool to handle this case.

That's why we have a function decorator ``minpy.core.convert_args``. If we apply
decorator ``minpy.core.convert_args`` on ``simple_add``, we can wrap the input into
MinPy's data type before passed to the function. Now the operation add is in MinPy's
namespace, and therefore enjoys MinPy's GPU acceleration if GPU is available. Now, we have

::

    import minpy.numpy as np
    from minpy.core import wraps
    @wraps()
    def simple_add(a, b):
        return a + b


Notes
^^^^^

1. No conversion will be performed for the return values.

2. Even if you forget to add ``@convert_args`` decorator when you are working with
NumPy data, the gradient solver will still get correct gradients. The only drawback
is some operations will be performed in NumPy's namespace under some rare conditions.

``@return_numpy``: Return NumPy Arrays
--------------------------------------

This is a simple wrapper in ``minpy.core`` which converts the MinPy output of a function to NumPy arrays.
