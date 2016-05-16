#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Registry for functions under the same symbol."""
from ..utils import log
from ..utils import common
import types
import logging

_logger = log.get_logger(__name__)

class DuplicateRegistryError(ValueError):
    pass

class Registry(object):
    """Registry for primitives under the same symbol."""

    def __init__(self):
        self._reg = {}

    def register(self, name, prim):
        """Register primitive.

        :param str name: Name of the primitive
        :param Primitive prim: Primitive

        :raises DuplicateRegistryError: Type already registered under the same
                                        name.
        """
        if name not in self._reg:
            self._reg[name] = {}
        if prim.type in self._reg[name]:
            raise DuplicateRegistryError(
                'Type "{}" for name "{}" has already existed'.format(prim.typestr, name))
        else:
            _logger.debug('Function "{}" registered with type {}'
                         .format(name, prim.typestr))
            self._reg[name][prim.type] = prim

    def has_name(self, name):
        return name in self._reg

    def exists(self, name, t):
        return name in self._reg and t in self._reg[name]

    def get(self, name, t):
        return self._reg[name][t]

    def iter_available_types(self, name, args_len, kwargs_keys):
        if name not in self._reg:
            return iter([])
        else:
            ret = []
            for i in self._reg[name].values():
                if i.gradable(args_len, kwargs_keys):
                    ret.append(i)
            return ret
