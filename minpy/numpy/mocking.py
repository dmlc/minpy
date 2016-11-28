#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mocking module class.

It is used to dispatch names to different implementations (primitives). The primitive is decided
by policy which could be specified by user.
"""
from __future__ import absolute_import

import importlib

from minpy.array_variants import variants
from minpy.dispatch.registry import Registry
from minpy.dispatch.policy import Policy, PreferMXNetPolicy
from minpy.dispatch.primitive_selector import PrimitiveSelector
from minpy.primitive import Primitive
from minpy.utils import log
import minpy


class Module(object):
    """Mocking module class for name dispatching.

    It will register primitives from :mod:`minpy.array_variant`.

    Parameters
    ----------
    old : dict
        A meta class including info such as name, path, etc.
    name : None, or str
        Second level name if specified.
    name_injector : dict, or dict-like object
        An optional dict provides manual dispatching
    """

    def __init__(self, old, name=None, name_injector={}):
        # Add module itself into global config
        minpy.Config['modules'].append(self)
        self._registry = Registry()
        self._policy = minpy.Config['default_policy']
        self._logger = log.get_logger(old['__name__'])
        self._logger.info('Initialize module: {}.'.format(old['__name__']))
        self._old = old
        self._name_injector = name_injector
        if len(name_injector) != 0:
            self._logger.info('Inject Name Injector {}'.format(name_injector.__name__))
        for vname in variants:
            if name is None:
                modname = 'minpy.array_variants.{}'.format(vname)
            else:
                modname = 'minpy.array_variants.{}.{}'.format(vname, name)
            mod = importlib.import_module(modname)
            self._logger.info('Importing from {}.'.format(modname))
            primitive_wrapper = lambda func, *args, **kwargs:\
                    Primitive(func, variants[vname], *args, **kwargs)
            # Register all primitives of the module.
            before = len(self._registry._reg)
            mod.register_primitives(self._registry, primitive_wrapper)
            self._logger.info('Got {} primitives from {}'.format(
                len(self._registry._reg) - before, modname))
            primitive_getter = lambda name: self._registry.get(name, variants[vname])
            # Define gradients of primitives.
            mod.def_grads(primitive_getter)
        self._logger.info('Import {} primitives'.format(
            len(self._registry._reg)))

    def set_policy(self, plc):
        """Set name dispatch policy.

        Parameters
        ----------
        plc
            New policy.
        """
        assert isinstance(
            plc, Policy), 'Need an instance of `minpy.dispatch.policy.Policy`.'
        self._policy = plc

    @property
    def policy(self):
        """Get policy of current module"""
        return self._policy

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
            return PrimitiveSelector(name, self._registry, self._policy)
        elif name in self._name_injector:
            return self._name_injector[name]
        elif name in self._old:
            self._logger.info(
                'No entry found for "{}" in registry, fallback.'.format(name))
            return self._old[name]
        else:
            raise AttributeError('Cannot find name "{}".'.format(name))


class NameInjector(object):
    """A proxy class dispatching given names into another class

    Parameters
    ----------
    dest_mod : module
        The target module of the name dispatch.
    name : str
        Name of the instance. Used for logger.
    injected_type : None, or iterable of types
        List of the types. Objects in the list will be registerd for the proxy.
    name_set : None, or iterable of types
        Set of the names registered for the proxy.
    exception : None, or dict
        A dictionary provides exceptions. The dictionary matches injected name
        to dispatched name (string to string).
    """

    def __init__(self,
                 dest_mod,
                 name='',
                 injected_type=None,
                 name_set=None,
                 exception=None):
        if len(name) != 0:
            name = '<' + name + '>'
        self.__name__ = name
        self._logger = log.get_logger(__name__)
        self._logger.info('Initialize Name Injector {}'.format(self.__name__))
        self._name_dict = {}
        dest_ns = dest_mod.__dict__
        if injected_type is not None:
            for name, obj in dest_ns.items():
                if type(obj) in injected_type:
                    self._name_dict[name] = obj
                    self._logger.debug(
                        'Add {} from {} to Name Injector {}'.format(
                            name, dest_mod.__name__, self.__name__))
        if name_set is not None:
            for name in name_set:
                self._name_dict[name] = dest_ns[name]
                self._logger.debug(
                    'Add {} from {}i to Name Injector {}'.format(
                        name, dest_mod.__name__, self.__name__))
        if exception is not None:
            exception = {k: dest_ns[v] for k, v in exception.items()}
            self._logger.debug('Update {} exceptions to Name Injector {}'.
                               format(len(exception), self.__name__))
            self._name_dict.update(exception)
        self._logger.info('Import {} objects from {} to Name Injector {}'.
                          format(len(self), dest_mod.__name__, self.__name__))

    def __len__(self):
        return len(self._name_dict)

    def __contains__(self, key):
        return key in self._name_dict

    def __getitem__(self, name):
        if name in self._name_dict:
            return self._name_dict[name]
        else:
            raise KeyError(name)
