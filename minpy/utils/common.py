#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common utility functions."""
import functools

def enforce_type(T, method=False):
    # pylint: disable= missing-docstring, invalid-name
    """Enforce argument types.

    Args:
        T: Type expected.
        method: Whether it is a method, skip first argument `self`.

    Returns:
        A decorator.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            def wrap_arg(i, arg):
                if method and i == 0:
                    return arg
                elif not isinstance(arg, T):
                    return T(arg)
                else:
                    return arg

            def wrap_kwarg(kwarg):
                if not isinstance(kwarg, T):
                    return T(kwarg)
                else:
                    return kwarg
            args_wrapped = [wrap_arg(i, arg) for i, arg in enumerate(args)]
            kwargs_wrapped = {
                k: wrap_kwarg(kwarg) for k, kwarg in kwargs.items()
            }
            return func(*args_wrapped, **kwargs_wrapped)
        return wrapped
    return decorator
