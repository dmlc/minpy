#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable= protected-access, invalid-name
"""A light-weight profiler
USAGE
    - set MINPROF_FLAG to enable the profiler
    - add @minprof before the function you want to profiling
    - use minprof(<func>) to wrap a function
    - use with(<some info>): ... to wrap a code snippet
"""
import argparse
import time
import os
import sys
from inspect import getframeinfo, stack
import six.moves.cPickle as pickle  # pylint: disable= import-error, no-name-in-module

MINPROF_FLAG = True
# Python 2/3 compatibility utils
# =================================================
PY3 = sys.version_info[0] == 3

if PY3:
    import builtins  # pylint: disable= import-error
    exec_ = getattr(builtins, "exec")
    del builtins
else:

    def exec_(_code_, _globs_=None, _locs_=None):
        # pylint: disable= exec-used, unused-argument
        """Execute code in a namespaces."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")
# =================================================


def label(code):
    """ Return a (filename, first_lineno, func_name) tuple for a given code
    object.
    """
    if isinstance(code, str):
        return ('~', 0, code)
    elif isinstance(code, tuple):
        return code
    else:
        return (code.co_filename, code.co_firstlineno, code.co_name)


class FuncCallStats(object):
    """ Object to encapsulate function-call profiling statistics.

    Attributes:
        timings : dict
            Mapping from (filename, first_lineno, function_name) of the profiled
            function to a list of (nhits, total_time) tuples for all function call.
            total_time is an integer in the native units of the timer.
    """

    def __init__(self, timings):
        self.timings = timings


def read_lines(filename, begin_lineno, end_lineno):
    """Read lines from file."""
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i >= begin_lineno and i < end_lineno:
                lines.append(line.rstrip())

    return lines


class FuncCallProfiler(object):
    """Profiler for function."""
    class Timer(object):
        """Record time class."""
        def __init__(self, profiler, info=None):
            self.profiler = profiler
            self.info = info
            self.filename = None
            self.begin_lineno = None
            self.end_lineno = None
            self.code_snippet = None
            self.begin_time = None
            self.end_time = None

        def __enter__(self):
            x = getframeinfo(stack()[1][0])
            self.filename = x.filename
            self.begin_lineno = x.lineno
            self.begin_time = time.time()

        def __exit__(self, *exc):
            self.end_time = time.time()

            x = getframeinfo(stack()[1][0])
            self.end_lineno = x.lineno
            self.code_snippet = read_lines(self.filename, self.begin_lineno,
                                           self.end_lineno)

            if self.info is None:
                code = (self.filename, self.begin_lineno,
                        self.code_snippet[0].strip() + '...')
            else:
                code = (self.filename, self.begin_lineno, self.info)
            if code not in self.profiler.code_map:
                self.profiler.code_map[code] = []
            timing = (1, self.begin_time, self.end_time)
            self.profiler.code_map[code].append(timing)

    def __init__(self, *funcs):
        self.code_map = {}
        self.enable_count = 0
        for func in funcs:
            self.add_function(func)

    def __call__(self, func=None):
        if func is None:
            return FuncCallProfiler.Timer(self)
        if isinstance(func, str):
            return FuncCallProfiler.Timer(self, info=func)
        self.add_function(func)
        wrapper = self.wrap_function(func)
        return wrapper

    def extract_code(self, func, stackdepth):
        # pylint: disable= no-self-use
        """Extract code from frame info."""
        try:
            code = func.__code__
        except AttributeError:
            frameinfo = getframeinfo(stack()[stackdepth][0])
            code = (frameinfo.filename, frameinfo.lineno, func.__name__)
        return code

    def add_function(self, func):
        """ Record function-call profiling information for the
        given Python function.
        """
        code = self.extract_code(func, 3)
        if code not in self.code_map:
            self.code_map[code] = []

    def wrap_function(self, func):
        """ Wrap a function to profile it.
        """
        # pylint: disable= missing-docstring
        def wrapper(*args, **kwargs):
            code = self.extract_code(func, 2)
            begin_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.time()
            self.code_map[code].append((1, begin_time, end_time))
            return result

        if MINPROF_FLAG is True:
            return wrapper
        else:
            return func

    def dump_stats(self, filename):
        """ Dump a representation of the data to a file as a pickled FuncCallStats
        object from `get_stats()`.
        """
        fstats = self.get_stats()
        with open(filename, 'wb') as f:
            pickle.dump(fstats, f, pickle.HIGHEST_PROTOCOL)

    def print_stats(self, stream=None):
        """ Show the gathered statistics.
        """
        pstats = self.get_stats()
        show_text(pstats.timings, stream=stream)

    def get_stats(self):
        """ Get a representation of the data to a file as a pickled FuncCallStats
        object.
        """
        # stats: {(filename, first_lineno, function_name) : (hits, time)}
        stats = {}
        for code in self.code_map:
            key = label(code)
            stats[key] = self.code_map[code]
        return FuncCallStats(stats)

    def runctx(self, cmd, globals_env, locals_env):
        # pylint: disable= no-self-use
        """ Profile a single executable statement in the given namespaces.
        """
        exec_(cmd, globals_env, locals_env)


def show_func(filename, first_lineno, func_name, timings, stream=None):
    """ Show results for a single function.
    """
    if stream is None:
        stream = sys.stdout

    nhits = 0
    total_time = 0.0
    for _, begin_time, end_time in timings:
        nhits += 1
        total_time += end_time - begin_time
    if nhits == 0:
        return
    else:
        percall = total_time / nhits
    template = '%10s %10.4f %10.4f    %-s:%s(%s)'
    if len(filename) > 40:
        filename = '...' + filename[-40:]
    text = template % (nhits, total_time * 1000, percall * 1000, filename,
                       first_lineno, func_name)
    stream.write(text)
    stream.write("\n")


def show_text(stats, stream=None):
    """ Show text for the given timings.
    """
    if stream is None:
        stream = sys.stdout
    stream.write('\nMINPROF_FLAG: %s\n' % MINPROF_FLAG)
    if MINPROF_FLAG:
        stream.write('\nFile: %s\n' % __file__)
        stream.write('Timer unit: %g s\n\n' % 1e-03)
        template = '%10.8s %10.8s %10.8s    %-s:%s(%s)::%s'
        header = template % ('ncalls', 'tottime', 'percall', 'filename',
                             'lineno', 'function', 'info')
        stream.write("\n")
        stream.write(header)
        stream.write("\n")
        for (fn, lineno, name), _ in sorted(stats.items()):
            show_func(fn, lineno, name, stats[fn, lineno, name], stream=stream)
        stream.write("\n")
        stream.write("\n")


def find_script(script_name):
    """ Find the script.
    If the input is not a file, then $PATH will be searched.
    """
    if os.path.isfile(script_name):
        return script_name
    path = os.getenv('PATH').split(os.pathsep)
    for d in path:
        if d == '':
            continue
        fn = os.path.join(dir, script_name)
        if os.path.isfile(fn):
            return fn

    sys.stderr.write('Could not find script %s\n' % script_name)
    raise SystemExit(1)


minprof = FuncCallProfiler()


def main(args=None):
    """ Main function as a module tool
    Usage
        -p program level
        -f function-call level
        -l line-by-line level
    """
    if args is None:
        args = sys.argv
    usage = "%prog scriptfile [arg] ..."
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument(
        '-f',
        '--function-call',
        action='store_true',
        help="Use the function-call profiler")
    parser.add_argument(
        '-o',
        '--outfile',
        default=None,
        help="Save stats to <outfile>")
    parser.add_argument(
        '-v',
        '--view',
        action='store_true',
        help="View the results of the profile in addition to saving it.")

    options, option_args = parser.parse_args()
    if not options.outfile:
        extension = 'fprof'
        options.outfile = '%s.%s' % (os.path.basename(option_args[0]),
                                     extension)
    script_file = find_script(option_args[0])
    sys.path.insert(0, os.path.dirname(script_file))
    try:
        try:
            execfile_ = execfile  # pylint: disable= unused-variable, undefined-variable
            minprof.runctx('execfile_(%r, globals())' %
                           (script_file, ), globals(), locals())
        except (KeyboardInterrupt, SystemExit):
            pass
    finally:
        minprof.dump_stats(options.outfile)
        print('Wrote profile results to %s' % options.outfile)
        if options.view:
            minprof.print_stats()

if __name__ == '__main__':
    main(sys.argv)
