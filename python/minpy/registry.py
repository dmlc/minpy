#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Registry for functions under the same symbol."""
from .utils import log
from .utils import common
from . import policy

logger = log.get_logger(__name__)


class FunctionType(common.AutoNumber):
    """Enumeration of types of functions."""
    NUMPY = ()
    MXNET = ()


class DuplicateRegistryError(ValueError):
    pass


class Registry(object):
    """Registry for functions under the same symbol."""

    def __init__(self):
        self._reg = {}

    def register(self, name, func, t):
        """Register function.

        Args:
            name: Name of the function.
            func: Function itself.
            t: Type of function.

        Raises:
            DuplicateRegistryError: Type already registered under the same
                name.
        """
        if name not in self._reg:
            self._reg[name] = {}
        if t in self._reg[name]:
            raise DuplicateRegistryError(
                'Type {} for name {} is already present'.format(t, name))
        logger.info('Function {} registered to {} with type {}'
                    .format(func, name, t))
        self._reg[name][t] = func

    def exists(self, name, t):
        return name in self._reg and t in self._reg[name]

    def get(self, name, t):
        return self._reg[name][t]

    def iter_available_types(self, name):
        if name not in self._reg:
            return iter([])
        else:
            return self._reg[name].keys()


global_registry = Registry()


def resolve_name(name, args, kwargs, policy=policy.global_policy,
                 registry=global_registry):
    """Resolve a function name.

    Args:
        name: Name of the function.
        args: Arguments.
        kwargs: Keyword arguments.
        policy: Resolving policy.
        registry: Registry for functions.

    Returns:
        A function after resolution.
    """
    t = policy.decide(name, args, kwargs)
    return registry.get(name, t)
