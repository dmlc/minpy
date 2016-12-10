"""Rules for rule based policy"""
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

_logger = log.get_logger(__name__)  # pylint: disable=invalid-name

# TODO: integrate this part into normal routine when MXNet fixes exception in
# Python.
# Currently MXNet doesn't throw exception raised in mshadow to Python. Avoid
# them by specifying a handcraft whitelist.
MXNET_SUPPORT_TYPES = {'float', 'float16', 'float32'}
MXNET_TYPE_COMPATIBLE_OPS = {
    'negative', 'add', 'subtract', 'multiply', 'divide', 'true_divide', 'mod',
    'power'
}
# These are MXNet ops that introduces potential issues for further computation.
MXNET_BLACKLIST_OPS = {'array'}


class RuleError(ValueError):
    """Error in rule processing"""
    pass


class Rules(object):
    """Rules interface.

    Different rule instances act like a single singleton.

    Parameters
    ----------
    loc : str
        Path to rule configuration file.
    save_config_atexit : bool
        True will save config after the program exits.
    """
    _rules = None
    _hash = None
    _env_var = '$MINPY_CONF'
    _conf_file = '.minpy_rules.conf'
    _loc = None
    _save_config_atexit = False

    def __init__(self, loc=None, save_config_atexit=False):
        self.__class__._loc = loc  # pylint: disable=protected-access
        if save_config_atexit and not self._save_config_atexit:
            self.__class__._save_config_atexit = True  # pylint: disable=protected-access
            atexit.register(self.save_rules_config)
        self.load_rules_config()

    @classmethod
    def _build_hash(cls):
        """Clear hash and rebuild hash by rules"""
        raise NotImplementedError()

    @classmethod
    def load_rules_config(cls, force=False):
        """Load rules configuration from configs and build hash.

        Find rule configuration at current directory, self._env_var, and user's
        root in order. Then load the config into corresponding class variable.
        Load empty rules if loading fails.

        Parameters
        ----------
        force : bool
            if True, force to load configuration.
        """
        # TODO: add package data through installation
        # http://peak.telecommunity.com/DevCenter/setuptools#non-package-data-files
        if cls._rules is None or force:
            config = None
            locs = [
                os.curdir,
                os.path.expandvars(cls._env_var),
                os.path.expanduser('~'),
            ]
            locs = [os.path.join(loc, cls._conf_file) for loc in locs]
            if cls._loc is not None:
                locs.insert(0, cls._loc)
            for filename in locs:
                try:
                    with open(filename) as config_file:
                        config = yaml.safe_load(config_file)
                    break
                except IOError:
                    pass
                except yaml.YAMLError:
                    _logger.warning('Find corrupted configuration at %s',
                                    filename)
            if config is None:
                _logger.error(
                    "Cannot find MinPy's rule configuration %s at %s.",
                    cls._conf_file, locs)
                config = {}
            else:
                _logger.info('Load and use rule configuration at %s', filename)  # pylint: disable=undefined-loop-variable
            cls._rules = config
            cls._build_hash()

    @property
    def name(self):
        """Return name of the policy"""
        return self.__class__.__name__

    @classmethod
    def save_rules_config(cls):
        '''Save rules configuration from configs and build hash.

        Save
        '''
        loc = cls._loc
        if loc is None:
            loc = os.environ.get(cls._env_var)
            if loc is None:
                loc = os.path.expanduser('~')
            loc = os.path.join(loc, cls._conf_file)
        with open(loc, 'w+') as config_file:
            yaml.safe_dump(cls._rules, config_file, default_flow_style=False)
        _logger.info('Rule %s saved to %s.', cls.__name__, loc)

    @classmethod
    def reset_rules(cls):
        """Reset rules.

        Delete all current rules. Also clear hash.
        """
        cls._rules = {}
        cls._hash = {}

    def allow(self, name, nspace, impl_type, args, kwargs):
        """Rule inquiry interface.

        Check if implementation is allowed.

        Parameters
        ----------
        name : str
            The dispatch name.
        nspace : str
            The namespace the dispatch name belongs to.
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
        # pylint: disable=too-many-arguments
        raise NotImplementedError()

    def add(self, name, nspace, impl_type, args, kwargs):
        """Rule registration interface.

        Register a new rule based on given info.

        Parameters
        ----------
        name : str
            The dispatch name.
        nspace : str
            The namespace of the dispatch name.
        impl_type : ArrayType
            The type of implementation.
        args : list
            The positional arguments passed to the primitive.
        kwargs : dict
            The keyword arguments passed to the primitive.
        """
        # pylint: disable=too-many-arguments
        raise NotImplementedError()


class Blacklist(Rules):
    """Blacklist rules for rule-based policy"""

    def allow(self, name, nspace, impl_type, args, kwargs):
        # pylint: disable=too-many-arguments
        if impl_type != ArrayType.MXNET:
            return True
        # For simplicity, this applies to all namespaces.
        if name in MXNET_BLACKLIST_OPS:
            _logger.debug(
                'Rule applies: %s is in internal MXNet op blacklist.', name)
            return False

        def _is_supported_array_type(var):
            if isinstance(var, Array):
                # TODO: simplify here when MXNet, NumPy .dtype behavior become
                # consistent
                return numpy.dtype(var.dtype).name in MXNET_SUPPORT_TYPES
            else:
                return True

        if nspace in self._hash and name in self._hash[nspace] and (
                self._hash[nspace][name] is None or self._get_arg_rule_key(
                    args, kwargs) in self._hash[nspace][name]):
            _logger.debug('Rule applies: block by auto-generated rule on %s.',
                          name)
            return False
        if name in MXNET_TYPE_COMPATIBLE_OPS:
            return True
        if not all(_is_supported_array_type(x) for x in args):
            _logger.debug(
                'Rule applies: contain unsupported type for MXNet op.')
            return False

        return True

    def add(self, name, nspace, impl_type, args, kwargs):
        # pylint: disable=too-many-arguments
        if impl_type != ArrayType.MXNET:
            raise RuleError('This rule only blacklists MXNet ops.')

        # Return type sequence
        type_seq = lambda args: [self._get_type_signiture(x) for x in args]

        self._rules.setdefault(nspace, {})
        self._rules[nspace].setdefault(name, [])
        self._hash.setdefault(nspace, {})
        self._hash[nspace].setdefault(name, set())
        if self._get_arg_rule_key(args,
                                  kwargs) not in self._hash[nspace][name]:
            entry = {'args': type_seq(args)}
            if len(kwargs) > 0:
                entry['kwargs'] = list(kwargs.keys())
            self._rules[nspace][name].append(entry)
            key = self._get_arg_rule_key(args, kwargs)
            self._hash[nspace][name].add(key)
            _logger.info('New rule %s added.', key)

    @classmethod
    def _build_hash(cls):
        cls._hash = {}
        for nspace, ns_rules in cls._rules.items():
            cls._hash[nspace] = {}
            for key, rules in ns_rules.items():
                cls._hash[nspace][key] = set()
                for elm in rules:
                    cls._hash[nspace][key].add('-'.join(elm[
                        'args']) + '+' + '-'.join(
                            sorted(elm.get('kwargs', []))))

    @staticmethod
    def _get_type_signiture(var):
        if isinstance(var, Array):
            return 'array_dim' + str(var.ndim)
        elif isinstance(var, Number):
            return type(var.val).__name__
        else:
            return type(var).__name__

    def _get_arg_rule_key(self, args, kwargs):
        arg_key = [self._get_type_signiture(x) for x in args]
        kwarg_key = sorted(kwargs.keys())
        return '-'.join(arg_key) + '+' + '-'.join(kwarg_key)

    @classmethod
    def query(cls, nspace, name):
        """Query the content of the rule by primitive name.

        Parameters
        ----------
        nspace
            The namespace of the given primitive.
        name : str
            Name of the primitive for query

        Returns
        -------
        str
            Return the rule content of primitive name.
        """
        ns_path = nspace.__name__
        if not hasattr(nspace, name):
            return '{} has no attribute {}.'.format(ns_path, name)
        if ns_path not in cls._rules or name not in cls._rules[ns_path]:
            return 'No rule for {} is found in {}.'.format(name, ns_path)
        else:
            from tabulate import tabulate
            rule_list = cls._rules[ns_path][name]
            rows = [(i + 1, ', '.join(rule.get('args', [])),
                     ', '.join(rule.get('kwargs', [])))
                    for i, rule in enumerate(rule_list)]
            table = tabulate(
                rows,
                headers=('No.', 'Type of Positional Args', 'Keyword Args'),
                tablefmt='grid')
            return 'Total: {} blacklist rules for primitive [{}]:\n'.format(
                len(rule_list), name) + table
