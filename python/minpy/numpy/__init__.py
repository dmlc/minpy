from __future__ import absolute_import

from .. import array_variants
from ..policy import default_policy

import importlib
from ..array_variants.numpy import *

def _minpy_import_array_pkg(pkg = None):
    for v in array_variants.variants:
        if pkg != None:
            importlib.import_module(pkg, __name__ + '.array_variants.' + v)
        else:
            importlib.import_module(v, __name__ + '.array_variants')

def resolve_name(name, args, kwargs, registry, policy=default_policy):
    """Resolve a function name.

    Args:
        name: Name of the function.
        args: Arguments.
        kwargs: Keyword arguments.
        registry: Registry for functions.
        policy: Resolving policy.

    Returns:
        A function after resolution.
    """
    preference = policy.decide(name, args, kwargs)
    available = registry.iter_available_types(name)
    if preference in available or len(available) == 0:
        return registry.get(name, preference)
    else:
        return registry.get(name, available[0])

#_minpy_import_array_pkg()
