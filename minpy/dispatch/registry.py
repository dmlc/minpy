#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Registry for functions under the same symbol."""
from ..utils import log

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)
# pylint: enable= invalid-name


class PrimitiveRegistryError(ValueError):
    """ Error during registering primitives """
    pass


class Registry(object):
    """ Registry for primitives. Primitives with the same name but with different implementation
    type will be registered in the same entry.
    """

    def __init__(self):
        self._reg = {}

    def register(self, name, prim):
        """Register primitive.

        :param name: Name of the primitive.
        :param prim: Primitive.
        :raises PrimitiveRegistryError: Type already registered under the same name.
        """
        if name not in self._reg:
            self._reg[name] = {}
        if prim.type in self._reg[name]:
            raise PrimitiveRegistryError(
                'Type "{}" for name "{}" has already existed'.format(
                    prim.typestr, name))
        else:
            _logger.debug(
                'Function "{}" registered with type {}'.format(
                    name, prim.typestr))
            self._reg[name][prim.type] = prim

    def has_name(self, name):
        """ Return whether the given name has been registered """
        return name in self._reg

    def exists(self, name, ptype):
        """ Return whether primitive exists under the given name and the given implementation type.
        """
        return name in self._reg and ptype in self._reg[name]

    def get(self, name, ptype):
        """ Get the primitive registered under the given name and the given implementation type.
        """
        return self._reg[name][ptype]

    def iter_available_types(self, name, bp_args, bp_kwargs):
        """Find primitives of the given name that have gradients defined for the arguments.

        :param str name: Primitive name.
        :param tuple bp_args: Positional arguments that need back propagation.
        :param tuple bp_kwargs: Keyword arguments that need back propagation.
        :return: Primitives that satisfy above requirements.
        """
        if name not in self._reg:
            return iter([])
        else:
            ret = []
            for prim in self._reg[name].values():
                if prim.gradable(bp_args, bp_kwargs):
                    ret.append(prim)
            return ret
