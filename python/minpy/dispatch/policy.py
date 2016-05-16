#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Policy for selecting appropriate function to call."""
from ..utils import log
from ..array_variants import ArrayType

_logger = log.get_logger(__name__)


class AmbiguousPolicyError(ValueError):
    pass

class Policy(object):
    """Policy interface """

    def decide(self, candidates, *args, **kwargs):
        raise AmbiguousPolicyError('Unimplemented')

class PreferMXNetPolicy(Policy):
    """Perfer using MXNet functions."""
    def decide(self, candidates, *args, **kwargs):
        if ArrayType.MXNET in map(lambda x: x.type, candidates):
            return ArrayType.MXNET
        else:
            return ArrayType.NUMPY

class OnlyNumpyPolicy(Policy):
    """Perfer using MXNet functions."""
    def decide(self, candidates, *args, **kwargs):
        if ArrayType.NUMPY in map(lambda x: x.type, candidates):
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
