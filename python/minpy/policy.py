#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Policy for selecting appropriate function to call."""
import itertools
import functools
import operator
from .utils import log
from . import array
from . import registry

logger = log.get_logger(__name__)

class AmbiguousPolicyError(ValueError):
    pass

class Policy(object):

    def decide(self, *args, **kwargs):
        raise AmbiguousPolicyError('Unimplemented')


class PreferMXNetPolicy(Policy):
    """Perfer using MXNet functions."""

    def decide(self, *args, **kwargs):
        if functools.reduce(operator.or_, map(
                lambda x: x.get_type() == array.ArrayType.NUMPY,
                itertools.chain(args, kwargs.values())), False):
            return registry.FunctionType.NUMPY
        else:
            return registry.FunctionType.MXNET

default_policy = Policy()

def resolve_name(name, args, kwargs, reg, policy=default_policy):
    """Resolve a function name.

    Args:
        name: Name of the function.
        args: Arguments.
        kwargs: Keyword arguments.
        reg: Registry for functions.
        policy: Resolving policy.

    Returns:
        A function after resolution.
    """
    preference = policy.decide(name, args, kwargs)
    available = reg.iter_available_types(name)
    if preference in available or len(available) == 0:
        return reg.get(name, preference)
    else:
        return reg.get(name, available[0])
