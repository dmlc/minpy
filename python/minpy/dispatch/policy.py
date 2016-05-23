#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= no-self-use
"""Policy for selecting appropriate function to call."""
from ..utils import log
from ..array_variants import ArrayType

#pylint: disable= invalid-name
_logger = log.get_logger(__name__)
#pylint: enable= invalid-name

class PrimitivePolicyError(ValueError):
    """ Error during choosing primitives """
    pass

class Policy(object):
    """Policy interface """
    def decide(self, candidates, *args, **kwargs):
        """ Primitive decision policy interface
        Args:
            candidates:
                A list of primitive objects
            args:
                The arguments passed to the primitive
            kwargs:
                The arguments passed to the primitive
        Return:
            Which implementation type will be used
        """
        raise PrimitivePolicyError('Unimplemented')

class PreferMXNetPolicy(Policy):
    """Perfer using MXNet functions."""
    def decide(self, candidates, *args, **kwargs):
        if ArrayType.MXNET in [x.type for x in candidates]:
            return ArrayType.MXNET
        else:
            return ArrayType.NUMPY

class OnlyNumpyPolicy(Policy):
    """Perfer using MXNet functions."""
    def decide(self, candidates, *args, **kwargs):
        if ArrayType.NUMPY in [x.type for x in candidates]:
            return ArrayType.NUMPY
        else:
            raise ValueError("Cannot find proper functions among: {}.".format(candidates))

def resolve_name(name, reg, plc, *args, **kwargs):
    """Resolve a function name.

    Args:
        name: Name of the function.
        args: Arguments.
        kwargs: Keyword arguments.
        reg: Registry for functions.
        plc: Resolving policy.

    Returns:
        A function after resolution.
    """
    args_len = len(args)
    kwargs_keys = list(kwargs.keys())
    available = reg.iter_available_types(name, args_len, kwargs_keys)
    preference = plc.decide(available, args, kwargs)
    return reg.get(name, preference)
