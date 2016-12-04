#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Registry for functions under the same symbol."""
from minpy.utils import log

_logger = log.get_logger(__name__)  # pylint: disable= invalid-name


class PrimitiveRegistryError(ValueError):
    """ Error during registering primitives """
    pass


class Registry(object):
    """ Registry for primitives. Primitives with the same name but with different implementation
    type will be registered in the same entry.
    """

    def __init__(self, namespace):
        self._reg = {}
        self._ns = namespace

    @property
    def nspace(self):
        """Return the namespace of the registry."""
        return self._ns

    def register(self, name, prim):
        """Register primitive.

        Parameters
        ----------
        name : str
            Name of the primitive.
        prim
            Registered primitive.

        Raises
        ------
        PrimitiveRegistryError
            Type already registered under the same name.
        """
        if name not in self._reg:
            self._reg[name] = {}
        if prim.type in self._reg[name]:
            raise PrimitiveRegistryError(
                'Type "{}" for name "{}" has already existed'.format(
                    prim.typestr, name))
        else:
            _logger.debug('Function "%s" registered with type %s', name,
                          prim.typestr)
            self._reg[name][prim.type] = prim

    def has_name(self, name):
        """Return whether the given name has been registered"""
        return name in self._reg

    def exists(self, name, ptype):
        """Return whether primitive exists under the given name and the given implementation type.
        """
        return name in self._reg and ptype in self._reg[name]

    def get(self, name, ptype):
        """Get the primitive registered under the given name and the given implementation type.
        """
        return self._reg[name][ptype]

    def iter_available_types(self, name, bp_args, bp_kwargs):
        """Find primitives of the given name that have gradients defined for the arguments.

        Parameters
        ----------
        name : str
            Primitive name.
        bp_args : tuple
            Positional arguments that need back propagation.
        bp_kwargs : tuple
            Keyword arguments that need back propagation.

        Returns
        -------
        Primitives that satisfy the requirements above.
        """
        # Just a redundant check. name must lay in self._reg by mocking.py.
        if name not in self._reg:
            return iter([])
        else:
            ret = []
            for prim in self._reg[name].values():
                if prim.gradable(bp_args, bp_kwargs):
                    ret.append(prim)
                else:
                    _logger.info('The %s implementation: %s has no gradient '
                                 'definition.', prim.typestr, prim)
            return ret
