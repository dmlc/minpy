#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import importlib
import logging

from ..utils import log
from ..dispatch import registry
from ..dispatch import policy
from ..array_variants import * # import all array_variants names
from .. import array

class DynamicLookupError(KeyError):
    pass

class Module(object):
    """Module level dynamic lookup."""

    def __init__(self, old, name=None):
        self._registry = registry.Registry()
        self._policy = policy.PreferMXNetPolicy()
        self._logger = log.get_logger(old['__name__'], logging.DEBUG)
        self._logger.info('Initialize module: {}'.format(old['__name__']))
        self._old = old
        for vname in variants:
            if name == None:
                modname = 'minpy.array_variants.{}'.format(vname)
            else:
                modname = 'minpy.array_variants.{}.{}'.format(vname, name)
            mod = importlib.import_module(modname)
            self._logger.info('Importing from {}'.format(modname))
            #TODO better wrapper?
            primitive_wrapper = lambda func : array.Primitive(func, variants[vname][1])
            # register all primitives of the module
            before = len(self._registry._reg)
            mod.register_primitives(self._registry, primitive_wrapper)
            self._logger.info('Got {} primitives from {}'.format(len(self._registry._reg) - before, modname))
            primitive_getter = lambda name : self._registry.get(name, variants[vname][1])
            # define gradients of primitives
            mod.def_grads(self._registry, primitive_getter)
        self._logger.info('Import {} primitives'.format(len(self._registry._reg)))
    
    def __getattr__(self, name):
        self._logger.debug('Look up name {}'.format(name))
        # Special members for internal use.
        if name == '__registry__':
            return self._registry
        elif self._registry.has_name(name):
            prim =  policy.resolve_name(name, self._registry, self._policy)
            self._logger.debug('Found primitive with name "{}" with type {}'.format(name, prim.typestr))
            return prim
        elif name in self._old:
            self._logger.info('No entry found for {} in registry, fallback to plain numpy'.format(name))
            return self._old[name]
        else:
            raise DynamicLookupError()
