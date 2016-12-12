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
from minpy.dispatch.policy import Policy
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

    def __init__(self, old, name=None, name_injector=None):
        # pylint: disable= protected-access, cell-var-from-loop
        # Add module itself into global config
        minpy.Config['modules'].append(self)
        self._registry = Registry(old['__name__'])
        self._policy = minpy.Config['default_policy']
        self._logger = log.get_logger(old['__name__'])
        self._logger.info('Initialize module: %s.', old['__name__'])
        self._old = old
        self._name_injector = name_injector if name_injector else {}
        if len(self._name_injector) != 0:
            self._logger.info('Inject Name Injector %s', name_injector.__name__)
        for vname in variants:
            vtype = variants[vname]
            if name is None:
                modname = 'minpy.array_variants.{}'.format(vname)
            else:
                modname = 'minpy.array_variants.{}.{}'.format(vname, name)
            mod = importlib.import_module(modname)
            self._logger.info('Importing from %s.', modname)
            primitive_wrapper = lambda func, *args, **kwargs:\
                    Primitive(func, vtype, *args, **kwargs)
            # Register all primitives of the module.
            before = len(self._registry._reg)
            mod.register_primitives(self._registry, primitive_wrapper)
            self._logger.info('Got %d primitives from %s',
                              len(self._registry._reg) - before, modname)
            primitive_getter = lambda name: self._registry.get(name, vtype)
            # Define gradients of primitives.
            mod.def_grads(primitive_getter)
        self._logger.info('Import %d primitives', len(self._registry._reg))
        self._set_attrs()
        # pylint: enable= protected-access, cell-var-from-loop

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

    def _set_attrs(self):
        """Set attributes for this module"""
        # The latter will override the former, so set attributes in reverse priority order
        for k, val in self._old.items():
            setattr(self, k, val)
        for k in self._name_injector.keys():
            setattr(self, k, self._name_injector[k])
        for k in self._registry._reg: # pylint: disable= protected-access
            fun = PrimitiveSelector(k, self)
            setattr(self, k, fun)
        if '__all__' in dir(self._old):
            setattr(self, '__all__', self._old.__all__)
        setattr(self, '__registry__', self._registry)


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
        # pylint: disable= too-many-arguments
        if len(name) != 0:
            name = '<' + name + '>'
        self.__name__ = name
        self._logger = log.get_logger(__name__)
        self._logger.info('Initialize Name Injector %s', self.__name__)
        self._name_dict = {}
        dest_ns = dest_mod.__dict__
        if injected_type is not None:
            for name, obj in dest_ns.items():
                if type(obj) in injected_type: # pylint: disable= unidiomatic-typecheck
                    self._name_dict[name] = obj
                    self._logger.debug(
                        'Add %s from %s to Name Injector %s',
                        name, dest_mod.__name__, self.__name__)
        if name_set is not None:
            for name in name_set:
                self._name_dict[name] = dest_ns[name]
                self._logger.debug(
                    'Add %s from %s to Name Injector %s',
                    name, dest_mod.__name__, self.__name__)
        if exception is not None:
            exception = {k: dest_ns[v] for k, v in exception.items()}
            self._logger.debug('Update %d exceptions to Name Injector %s',
                               len(exception), self.__name__)
            self._name_dict.update(exception)
        self._logger.info('Import %d objects from %s to Name Injector %s',
                          len(self), dest_mod.__name__, self.__name__)
        # pylint: enable= too-many-arguments

    def keys(self):
        """Return all injected names."""
        return self._name_dict.keys()

    def __len__(self):
        return len(self._name_dict)

    def __contains__(self, key):
        return key in self._name_dict

    def __getitem__(self, name):
        if name in self._name_dict:
            return self._name_dict[name]
        else:
            raise KeyError(name)
