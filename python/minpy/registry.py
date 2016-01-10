#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Registry for functions under the same symbol."""
import enum
from .utils import log

logger = log.get_logger(__name__)


class AutoNumber(enum.Enum):

    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class FunctionType(AutoNumber):
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
        log.info('Function {} registered to {} with type {}'
                 .format(func, name, t))
        self._reg[name][t] = func

    def exists(self, name, t):
        return name in self._reg and t in self._reg[name]

    def get(self, name, t):
        return self._reg[name][t]
