#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= no-self-use
"""Policy for selecting appropriate function to call."""
from __future__ import absolute_import
from __future__ import print_function

import functools
import minpy
from minpy.array import Value
from minpy.array_variants import ArrayType
from minpy.utils import log

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)
# pylint: enable= invalid-name


class PrimitivePolicyError(ValueError):
    """Error during choosing primitives."""
    pass

class Policy(object):
    """Policy interface."""

    def decide(self, candidates, args, kwargs):
        """Primitive decision policy interface.

        :param list candidates: A list of primitive objects.
        :param list args: The positional arguments passed to the primitive.
        :param dict kwargs: The keyword arguments passed to the primitive.
        :return: Which implementation type will be used.
        """
        raise NotImplementedError()

    @property
    def name(self):
        return type(self).__name__

    def __enter__(self):
      self._old_policy = minpy.Config['default_policy']
      minpy.set_global_policy(self)
      return self

    def __exit__(self, ptype, value, trace):
      minpy.set_global_policy(self._old_policy)


class PreferMXNetPolicy(Policy):
    """ Prefer using MXNet functions. Return None if no required function. """

    def decide(self, candidates, args, kwargs):
        possible_impl = set(x.type for x in candidates)
        if ArrayType.MXNET in possible_impl:
            return ArrayType.MXNET
        elif ArrayType.NUMPY in possible_impl:
            return ArrayType.NUMPY
        else:
            return None

class OnlyNumPyPolicy(Policy):
    """ Only use NumPy functions. Return None if no required function. """

    def decide(self, candidates, args, kwargs):
        if ArrayType.NUMPY in tuple(x.type for x in candidates):
            return ArrayType.NUMPY
        else:
            return None


class OnlyMXNetPolicy(Policy):
    """ Only use MXNet functions. Return None if no required function. """

    def decide(self, candidates, args, kwargs):
        if ArrayType.MXNET in tuple(x.type for x in candidates):
            return ArrayType.MXNET
        else:
            return None


def resolve_name(name, reg, plc, args, kwargs):
    """Resolve a function name.

    :param str name: Name of the function.
    :param reg: Registry for functions.
    :param Policy plc: Resolving policy.
    :param tuple args: Positional arguments.
    :param dict kwargs: Keyword arguments.
    :return: A function after resolution.
    """
    def fst(t):
        x, _ = t
        return x
    bp_args = tuple(map(fst, filter(lambda x: isinstance(
        x[1], Value) and x[1].marked_for_bp, enumerate(args))))
    bp_kwargs = tuple(map(fst, filter(lambda x: isinstance(
        x[1], Value) and x[1].marked_for_bp, kwargs.items())))
    available = reg.iter_available_types(name, bp_args, bp_kwargs)
    preference = plc.decide(available, args, kwargs)
    if preference is None:
        if len(bp_args) == len(bp_kwargs) == 0:
            raise PrimitivePolicyError(
                "Cannot find implementation for function: {}() under "
                "policy: {}.".format(name, plc.name))
        else:
            raise PrimitivePolicyError(
                "Cannot find function with proper gradient implementation for "
                ": {}() under policy: {}.".format(name, plc.name))
    return reg.get(name, preference)


def wrap_policy(policy):
    """Wrap a function to use specific policy

    Parameters
    ----------
    policy : Policy
        A MinPy Policy Instance

    Returns
    -------
    A wrapped function running under specific policy
    """
    def policy_decorator(func):
        # pylint: disable= missing-docstring
        @functools.wraps(func)
        def policy_wrapper(*args, **kwargs):
            old_policy = minpy.Config['default_policy']
            minpy.set_global_policy(policy)
            result = func(*args, **kwargs)
            minpy.set_global_policy(old_policy)
            return result
        return policy_wrapper
        # pylint: enable= missing-docstring
    return policy_decorator
