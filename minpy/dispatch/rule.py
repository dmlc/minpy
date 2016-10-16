"""Rules for rule-based policy"""
from __future__ import absolute_import
from __future__ import print_function

import os
import atexit
import yaml
import numpy
from minpy.array_variants import ArrayType
from minpy.array import Array
from minpy.array import Number
from minpy.utils import log

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)
# pylint: enable= invalid-name

# TODO: integrate this part into normal routine when MXNet fixes exception in
# Python.
# Currently MXNet doesn't throw exception raised in mshadow to Python. Avoid
# them by specifying a handcraft whitelist.
mxnet_support_types = {'float', 'float16', 'float32'}
mxnet_type_compatible_ops = {'negative', 'add', 'subtract', 'multiply',
                             'divide', 'true_divide', 'mod', 'power'}
# These are MXNet ops that introduces potential issues for further computation.
mxnet_blacklist_ops = {'array'}


class RuleError(ValueError):
    """Error in rule processing"""
    pass


class Rules(object):
    """Rules interface.

    Parameters
    ----------
    loc : str
        Path to rule configuration file.
    save_config_atexit : bool
        True will save config after the program exits.
    """

    def __init__(self, *args, **kwargs):
        self._rules = None
        self._conf_file = '.minpy_rules.conf'
        self._env_var = '$MINPY_CONF'
        self._rules = None
        self._hash = None
        self._save_config_atexit = False
        self._loc = None
        self(*args, **kwargs)

    def __call__(self, loc=None, save_config_atexit=False):
        if save_config_atexit:
            self._register_save_atexit()
        if not self._loc or loc:
            self._loc = loc
            self.load_rules_config()
        return self

    def _register_save_atexit(self):
        """Register a saving function when program exits."""
        if not self._save_config_atexit:
            self._save_config_atexit = True
            atexit.register(self.save_rules_config)

    def _build_hash(self):
        """Clear hash and rebuild hash by rules"""
        raise NotImplementedError()

    def load_rules_config(self):
        """Load rules configuration from configs and build hash.

        Find rule configuration at current directory, self._env_var, and user's
        root in order. Then load the config into corresponding class variable.
        Load empty rules if loading fails. Configuration path will be set to
        the file first found if loc is not specified.
        """
        # TODO: add package data through installation
        # http://peak.telecommunity.com/DevCenter/setuptools#non-package-data-files
        loaded = False
        locs = [os.curdir, os.path.expandvars(self._env_var),
                os.path.expanduser('~')]
        locs = [os.path.join(loc, self._conf_file) for loc in locs]
        if self._loc is not None:
            locs.insert(0, self._loc)
        for filename in locs:
            try:
                with open(filename) as f:
                    config = yaml.safe_load(f)
                self._rules = config
                self._build_hash()
                loaded = True
                if self._loc is None:
                    self._loc = filename
                break
            except IOError:
                pass
            except Exception:
                _logger.warn('Find corrupted configuration at %s', filename)
        if not loaded:
            if self._loc is None:
                self._loc = locs[-1]
            _logger.error(
                "Cannot find MinPy's rule configuration %s at %s. Create new config file at %s.",
                self._conf_file, locs, self._loc)
            self._rules = {}
            self._build_hash()
            self.save_rules_config()
        else:
            _logger.info('Load and use rule configuration at %s', self._loc)

    @property
    def name(self):
        """Return name of the policy"""
        return self.__class__.__name__

    def save_rules_config(self):
        '''Save rules configuration from configs.

        Must ensure self._loc is set before calling this function.
        '''
        loc = self._loc
        assert loc is not None, "The config saving path is None."
        with open(loc, 'w+') as f:
            yaml.safe_dump(self._rules, f, default_flow_style=False)
        _logger.info('Rule %s saved to %s.', self.name, loc)

    def reset_rules(self):
        """Reset rules.

        Delete all current rules. Also clear hash.
        """
        self._rules = {}
        self._hash = {}

    def allow(self, name, impl_type, args, kwargs):
        """Rule inquiry interface.

        Check if implementation is allowed.

        Parameters
        ----------
        name : str
            The dispatch name.
        impl_type : ArrayType
            The type of implementation.
        args : list
            The positional arguments passed to the primitive.
        kwargs : dict
            The keyword arguments passed to the primitive.

        Returns
        -------
        bool
            True if implementation is allowed; False otherwize.
        """
        raise NotImplementedError()

    def add(self, name, impl_type, args, kwargs):
        """Rule registration interface.

        Register a new rule based on given info.

        Parameters
        ----------
        name : str
            The dispatch name.
        impl_type : ArrayType
            The type of implementation.
        args : list
            The positional arguments passed to the primitive.
        kwargs : dict
            The keyword arguments passed to the primitive.
        """
        raise NotImplementedError()


class Blacklist(Rules):
    """Blacklist rules for rule-based policy"""

    def allow(self, name, impl_type, args, kwargs):
        if impl_type != ArrayType.MXNET:
            return True
        if name in mxnet_blacklist_ops:
            _logger.debug(
                'Rule applies: %s is in internal MXNet op blacklist.', name)
            return False

        def is_supported_array_type(x):
            if isinstance(x, Array):
                # TODO: simplify here when MXNet, NumPy .dtype behavior become
                # consistent
                return numpy.dtype(x.dtype).name in mxnet_support_types
            else:
                return True

        if name in self._hash and (
                self._hash[name] is None or
                self._get_arg_rule_key(args, kwargs) in self._hash[name]):
            _logger.debug('Rule applies: block by auto-generated rule on %s.',
                          name)
            return False
        if name in mxnet_type_compatible_ops:
            return True
        if not all(is_supported_array_type(x) for x in args):
            _logger.debug(
                'Rule applies: contain unsupported type for MXNet op.')
            return False

        return True

    def add(self, name, impl_type, args, kwargs):
        if impl_type != ArrayType.MXNET:
            raise RuleError('This rule only blacklists MXNet ops.')

        # Return type sequence
        type_seq = lambda args: [self._get_type_signiture(x) for x in args]

        self._rules.setdefault(name, [])
        self._hash.setdefault(name, set())
        if self._get_arg_rule_key(args, kwargs) not in self._hash[name]:
            entry = {'args': type_seq(args)}
            if len(kwargs) > 0:
                entry['kwargs'] = list(kwargs.keys())
            self._rules[name].append(entry)
            key = self._get_arg_rule_key(args, kwargs)
            self._hash[name].add(key)
            _logger.info('New rule %s added.', key)

    def _build_hash(self):
        self._hash = {}
        for k, v in self._rules.items():
            self._hash[k] = set()
            for x in v:
                self._hash[k].add('-'.join(x['args']) + '+' + '-'.join(
                    sorted(x.get('kwargs', []))))

    def _get_type_signiture(self, x):
        if isinstance(x, Array):
            return 'array_dim' + str(x.ndim)
        elif isinstance(x, Number):
            return type(x.val).__name__
        else:
            return type(x).__name__

    def _get_arg_rule_key(self, args, kwargs):
        arg_key = [self._get_type_signiture(x) for x in args]
        kwarg_key = sorted(kwargs.keys())
        return '-'.join(arg_key) + '+' + '-'.join(kwarg_key)
