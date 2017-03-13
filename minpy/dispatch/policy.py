#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Policy for selecting appropriate function to call."""
from __future__ import absolute_import
from __future__ import print_function

import functools
from collections import defaultdict
import minpy
from .. import tape
from ..array_variants import ArrayType
from ..utils import log
from .rule import Blacklist

_logger = log.get_logger(__name__)  # pylint: disable=invalid-name


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

    def __init__(self):
        self._mxnet_op_stat = defaultdict(int)
        self._numpy_op_stat = defaultdict(int)
        self._old_policy = None

    def decide(self, candidates):
        """Primitive decision policy interface.

        Note that this method is only used for default resolve_call.

        Parameters
        ----------
        candidates : list
            A list of primitive objects.

        Returns
        -------
        ArrayType or None
            The implementation type decided by the policy.
        """
        raise NotImplementedError()

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
        available = Policy._available_prims(name, reg, args, kwargs)
        preference = self.decide(available)
        if preference == ArrayType.MXNET:
            self._mxnet_op_stat[name] += 1
        elif preference == ArrayType.NUMPY:
            self._numpy_op_stat[name] += 1
        elif preference is None:
            raise PrimitivePolicyError(name, self.name)
        prim = reg.get(name, preference)
        _logger.debug('Found primitive "%s" with type %s.', name, prim.typestr)
        return prim.call(args, kwargs)

    def show_op_stat(self):
        """Print policy dispatch statistics."""
        mxnet_op_cnt = 0
        numpy_op_cnt = 0

        print('--------Op Dispatch Statistics Start--------')
        print('MXNET op called times:')
        for k, val in self._mxnet_op_stat.items():
            print(' {} : {}'.format(k, val))
            mxnet_op_cnt += val
        print('NUMPY op called times:')
        for k, val in self._numpy_op_stat.items():
            print(' {} : {}'.format(k, val))
            numpy_op_cnt += val
        total_cnt = mxnet_op_cnt + numpy_op_cnt
        if total_cnt > 0:
            print('Total Dispatch Proportion: {:.1%} in MXNet, {:.1%} in NumPy'.format(
                float(mxnet_op_cnt) / total_cnt,
                float(numpy_op_cnt) / total_cnt))
        print('--------Op Dispatch Statistics End--------')

    @property
    def name(self):
        """Return policy name"""
        return type(self).__name__

    @staticmethod
    def _available_prims(name, reg, args, kwargs):
        """Return a list of available primitives"""

        current_tape = tape.global_tape()

        bp_args = tuple(i for i, arg in enumerate(args)
                        if (hasattr(arg, 'is_marked_for_bp') and
                            arg.is_marked_for_bp(current_tape)))
        bp_kwargs = tuple(k for k, arg in kwargs.items()
                          if (hasattr(arg, 'is_marked_for_bp') and
                              arg.is_marked_for_bp(current_tape)))
        available = reg.iter_available_types(name, bp_args, bp_kwargs)
        return available


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

    # pylint: disable=abstract-method

    def __init__(self, gen_rule=False, append_rule=True, loc=None):
        super(AutoBlacklistPolicy, self).__init__()
        self._gen_rule = gen_rule
        self._rules = Blacklist(loc=loc, save_config_atexit=gen_rule)
        if gen_rule and not append_rule:
            self._rules.reset_rules()

    def resolve_call(self, name, reg, args, kwargs):
        def _get_result(impl_type):
            prim = reg.get(name, impl_type)
            return prim.call(args, kwargs)

        available = Policy._available_prims(name, reg, args, kwargs)
        possible_impl = set(x.type for x in available)
        if ArrayType.MXNET in possible_impl and self._rules.allow(
                name, reg.nspace, ArrayType.MXNET, args, kwargs):
            if self._gen_rule:
                try:
                    _logger.debug(
                        'Try primitive %s with MXNet implementation.', name)
                    res = _get_result(ArrayType.MXNET)
                    self._mxnet_op_stat[name] += 1
                    return res
                except Exception as err:  # pylint: disable=broad-except
                    if ArrayType.NUMPY in possible_impl:
                        _logger.info(
                            'Error occurs. Try primitive %s with NumPy implementation',
                            name)
                        self._rules.add(name, reg.nspace, ArrayType.MXNET, args, kwargs)
                        self._numpy_op_stat[name] += 1
                        return _get_result(ArrayType.NUMPY)
                    else:
                        raise err
            else:
                _logger.debug('Execute primitive %s with MXNet implementation',
                              name)
                self._mxnet_op_stat[name] += 1
                return _get_result(ArrayType.MXNET)
        elif ArrayType.NUMPY in possible_impl:
            _logger.debug('Execute primitive %s with NumPy implementation',
                          name)
            self._numpy_op_stat[name] += 1
            return _get_result(ArrayType.NUMPY)
        else:
            raise PrimitivePolicyError(name, self.name)

    def save_rules(self):
        """Save rules by rule's setting"""
        self._rules.save_rules_config()

    def query(self, nspace, name):
        """Query the content of the rule by primitive name in blacklist.

        Parameters
        ----------
        nspace
            The namespace of the given primitive. It is not a string.
        name : str
            Name of the primitive for query

        Returns
        -------
        str
            Return the rule content of primitive name.
        """
        return self._rules.query(nspace, name)


class PreferMXNetPolicy(Policy):
    """ Prefer using MXNet functions. Return None if no required function. """

    def decide(self, candidates):
        possible_impl = set(x.type for x in candidates)
        if ArrayType.MXNET in possible_impl:
            return ArrayType.MXNET
        elif ArrayType.NUMPY in possible_impl:
            return ArrayType.NUMPY
        else:
            return None


class OnlyNumPyPolicy(Policy):
    """ Only use NumPy functions. Return None if no required function. """

    def decide(self, candidates):
        if ArrayType.NUMPY in set(x.type for x in candidates):
            return ArrayType.NUMPY
        else:
            return None


class OnlyMXNetPolicy(Policy):
    """ Only use MXNet functions. Return None if no required function. """

    def decide(self, candidates):
        if ArrayType.MXNET in set(x.type for x in candidates):
            return ArrayType.MXNET
        else:
            return None


def wrap_policy(plc):
    """Decorate a function to use specific policy

    Parameters
    ----------
    plc : str or Policy object
        The policy to be used.

    Returns
    -------
    A wrapped function running under specific policy
    """

    def policy_decorator(func):
        # pylint: disable= missing-docstring
        @functools.wraps(func)
        def policy_wrapper(*args, **kwargs):
            old_policy = minpy.Config['default_policy']
            minpy.set_global_policy(plc)
            result = func(*args, **kwargs)
            minpy.set_global_policy(old_policy)
            return result

        return policy_wrapper

    return policy_decorator

def create(plc):
    """Create policy object.

    Parameters
    ----------
    plc : str or Policy object
        Name of the policy to be created.

    Returns
    -------
        Created policy object.
    """
    if isinstance(plc, Policy):
        return plc
    elif plc == 'prefer_mxnet':
        return PreferMXNetPolicy()
    elif plc == 'only_numpy':
        return OnlyNumPyPolicy()
    elif plc == 'only_mxnet':
        return OnlyMXNetPolicy()
    else:
        raise TypeError('Unknown policy name %s' % plc)
