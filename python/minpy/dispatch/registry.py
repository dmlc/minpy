#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Registry for functions under the same symbol."""
from ..utils import log
from ..utils import common
import types

logger = log.get_logger(__name__)


class FunctionType(common.AutoNumber):
    """Enumeration of types of functions.

    Semantically this is different from :class:`..array.ArrayType`,
    but for now one data type corresponds to one function type.
    """
    NUMPY = ()
    MXNET = ()


class DuplicateRegistryError(ValueError):
    pass


class Registry(object):
    """Registry for functions under the same symbol."""

    _reg = {}

    def register(self, name: str, func: types.FunctionType, t: FunctionType):
        """Register function.

        :param str name: Name of the function.
        :param function func: Function itself.
        :param FunctionType t: Type of function.

        :raises DuplicateRegistryError: Type already registered under the same
                                        name.
        """
        if name not in self._reg:
            self._reg[name] = {}
        if t in self._reg[name]:
            raise DuplicateRegistryError(
                'Type {} for name {} is already present'.format(t, name))
        else:
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
