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
from .rule import Blacklist

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)

# pylint: enable= invalid-name


class PrimitivePolicyError(ValueError):
    """Error during choosing primitives.

    Parameters
    ----------
    name : str
        Name waiting for dispatch.
    policy_name : str
        Name of the policy in which the error occurs.
    """

    def __init__(self, name, policy_name):
        super(PrimitivePolicyError,
              self).__init__("Cannot find implementation for function: {}() "
                             "under policy: {}. Maybe lack of gradient "
                             "implementation?".format(name, policy_name))


class Policy(object):
    """Policy interface."""

    def decide(self, candidates, args, kwargs):
        """Primitive decision policy interface.

        Parameters
        ----------
        candidates : list
            A list of primitive objects.
        args : list
            The positional arguments passed to the primitive.
        kwargs : dict
            The keyword arguments passed to the primitive.

        Returns
        -------
        ArrayType or None
            The implementation type decided by the policy.
        """
        raise NotImplementedError()

    @property
    def name(self):
        """Return policy name"""
        return type(self).__name__

    def __enter__(self):
        self._old_policy = {}
        for mod in minpy.Config['modules']:
            self._old_policy[mod] = mod.policy
            mod.set_policy(self)
        return self

    def __exit__(self):
        for mod, plc in self._old_policy.items():
            mod.set_policy(plc)

    @staticmethod
    def _available_prims(name, reg, args, kwargs):
        """Return a list of available primitives"""

        def fst(t):
            x, _ = t
            return x

        bp_args = tuple(
            map(fst, filter(
                lambda x: isinstance(x[1], Value) and x[1].marked_for_bp,
                enumerate(args))))
        bp_kwargs = tuple(
            map(fst, filter(
                lambda x: isinstance(x[1], Value) and x[1].marked_for_bp,
                kwargs.items())))
        available = reg.iter_available_types(name, bp_args, bp_kwargs)
        return available

    def resolve_call(self, name, reg, args, kwargs):
        """Resolve a function call.

        Parameters
        ----------
        name : str
            Name of the function.
        reg
            Registry for functions.
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        Result from appropriate function call.
        """
        available = self._available_prims(name, reg, args, kwargs)
        preference = self.decide(available, args, kwargs)
        if preference is None:
            raise PrimitivePolicyError(name, self.name)
        prim = reg.get(name, preference)
        _logger.debug('Found primitive "{}" with type {}.'.format(
            name, prim.typestr))
        return prim(*args, **kwargs)


class AutoBlacklistPolicy(Policy):
    """Automatically dispatch ops to MXNet impl by provided config.

    Note: different instances of the rule class act as a single singleton.

    Parameters
    ----------
    gen_rule : bool
        If False, use loaded rules to decide. Otherwise, dynamically add new
        rules and save to rule files.
    append_rule : bool
        If True, append new rules to loaded rules. Otherwise, start from
        scratch.
    loc : str
        Path to rule configuration file.
    """

    def __init__(self, gen_rule=False, append_rule=True, loc=None):
        self._gen_rule = gen_rule
        self._rules = Blacklist(loc=loc, save_config_atexit=gen_rule)
        if gen_rule and not append_rule:
            self._rules.reset_rules()

    def resolve_call(self, name, reg, args, kwargs):
        def get_result(impl_type):
            prim = reg.get(name, impl_type)
            return prim(*args, **kwargs)

        available = self._available_prims(name, reg, args, kwargs)
        possible_impl = set(x.type for x in available)
        if ArrayType.MXNET in possible_impl and self._rules.allow(
                name, ArrayType.MXNET, args, kwargs):
            if self._gen_rule:
                try:
                    _logger.debug('Try primitive {} with MXNet '
                                  'implementation.'.format(name))
                    return get_result(ArrayType.MXNET)
                except Exception as err:
                    if ArrayType.NUMPY in possible_impl:
                        _logger.info('Error occurs. Try primitive {} with '
                                     'NumPy implementation'.format(name))
                        self._rules.add(name, ArrayType.MXNET, args, kwargs)
                        return get_result(ArrayType.NUMPY)
                    else:
                        raise err
            else:
                _logger.debug('Execute primitive {} with '
                              'MXNet implementation'.format(name))
                return get_result(ArrayType.MXNET)
        elif ArrayType.NUMPY in possible_impl:
            _logger.debug('Execute primitive {} with '
                          'NumPy implementation'.format(name))
            return get_result(ArrayType.NUMPY)
        else:
            raise PrimitivePolicyError(name, self.name)

    def save_rules(self):
        """Save rules by rule's setting"""
        self._rules.save_rules_config()


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
