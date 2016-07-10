#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mocking module class.

It is used to dispatch names to different implementations (primitives). The primitive is decided
by policy which could be specified by user.
"""
from __future__ import absolute_import

import importlib

from ..utils import log
from ..dispatch import registry
from ..dispatch import policy
from ..dispatch import primitive_selector
from ..array_variants import variants
from .. import array


class Module(object):
    """Mocking module class for name dispatching.
    
    It will register primitives from :mod:`minpy.array_variant`.
    """

    def __init__(self, old, name=None):
        self._registry = registry.Registry()
        self._policy = policy.PreferMXNetPolicy()
        self._logger = log.get_logger(old['__name__'])
        self._logger.info('Initialize module: {}.'.format(old['__name__']))
        self._old = old
        for vname in variants:
            if name is None:
                modname = 'minpy.array_variants.{}'.format(vname)
            else:
                modname = 'minpy.array_variants.{}.{}'.format(vname, name)
            mod = importlib.import_module(modname)
            self._logger.info('Importing from {}.'.format(modname))
            primitive_wrapper = lambda func, *args, **kwargs:\
                    array.Primitive(func, variants[vname], *args, **kwargs)
            # Register all primitives of the module.
            before = len(self._registry._reg)
            mod.register_primitives(self._registry, primitive_wrapper)
            self._logger.info('Got {} primitives from {}'.format(
                len(self._registry._reg) - before, modname))
            primitive_getter = lambda name: self._registry.get(name, variants[vname])
            # Define gradients of primitives.
            mod.def_grads(self._registry, primitive_getter)
        self._logger.info('Import {} primitives'.format(len(
            self._registry._reg)))

    def set_policy(self, plc):
        """Set name dispatch policy.

        :param plc: New policy.
        """
        assert isinstance(
            plc, policy.Policy), 'Need an instance of `minpy.dispatch.Policy`.'
        self._policy = plc

    def __getattr__(self, name):
        """Fetch attributes from this module.
        
        If the name is contained in the primitive registry,
        it will return a primitive selector for further name dispatching.

        :param name: Name of attribute.
        :return: Primitive selector.
        :raises AttributeError: Cannot find attribute.
        """
        # Special members for internal use.
        if name == '__registry__':
            return self._registry
        elif name == '__all__':
            return self._old.__all__
        elif self._registry.has_name(name):
            return primitive_selector.PrimitiveSelector(name, self._registry,
                                                        self._policy)
        elif name in self._old:
            self._logger.info(
                'No entry found for "{}" in registry, fallback.'.format(name))
            return self._old[name]
        else:
            raise AttributeError('Cannot find name "{}".'.format(name))
