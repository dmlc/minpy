#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Policy for selecting appropriate function to call."""
import itertools
import functools
import operator
from .utils import log
from . import array
from . import registry

logger = log.get_logger(__name__)


class AmbiguousPolicyError(ValueError):
    pass


class Policy(object):

    def decide(self, *args, **kwargs):
        raise AmbiguousPolicyError('Unimplemented')


class PreferMXNetPolicy(Policy):
    """Perfer using MXNet functions."""

    def decide(self, *args, **kwargs):
        if functools.reduce(operator.or_, map(
                lambda x: x.get_type() == array.ArrayType.NUMPY,
                itertools.chain(args, kwargs.values())), False):
            return registry.FunctionType.NUMPY
        else:
            return registry.FunctionType.MXNET


default_policy = Policy()
