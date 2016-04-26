#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# TODO: nest, recursive
# TODO: GPU


try:
    import cPickle as pickle
except ImportError:
    import pickle

import functools
import optparse
import time
import os
import sys

# Python 2/3 compatibility utils
# =================================================
PY3 = sys.version_info[0] == 3

if PY3:
    import builtins
    exec_ = getattr(builtins, "exec")
    del builtins
else:
    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespaces."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = grame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")
# =================================================

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def label(code):
    """ Return a (filename, first_lineno, func_name) tuple for a given code
    object.
    """
    if isinstance(code, str):
        return ('~', 0, code)
    else:
        return (code.co_filename, code.co_firstlineno, code.co_name)

class FuncCallStats(object):
    """ Object to encapsulate function-call profiling statistics.

    Attributes
    ----------
    timings : dict
        Mapping from (filename, first_lineno, function_name) of the profiled
        function to a list of (nhits, total_time) tuples for all function call.
        total_time is an integer in the native units of the timer.

    TODO: uint : float
        The number of seconds per timer unit.
    """
    def __init__(self, timings):
        self.timings = timings
        # self.unit = unit


class FuncCallProfiler():
    def __init__(self, *funcs):
        # logger.info("prof init")
        # self.functions = []
        self.code_map = {}
        self.enable_count = 0
        for func in funcs:
            self.add_function(func)

    def add_function(self, func):
        """ Record function-call profiling information for the
        given Python function.
        """
        # logger.info("prof add function")
        try:
            code = func.__code__
            #print 'add code: ', code
        except AttributesError:
            import warnings
            wranings.warn("Could not extract a code object for the object %r" % (func, ))
            return
        if code not in self.code_map:
            self.code_map[code] = []

    # def __enter__(self):
    #     # do nothing now
    #     logger.info("prof enter")
    #     self.enable_by_count()

    # def __exit__(self):
    #     # do nothing now
    #     logger.info("prof exit")
    #     self.diable_by_count()

    def __call__(self, func):
        # logger.info("prof __call__")
        self.add_function(func)
        wrapper = self.wrap_function(func)
        return wrapper

    def wrap_function(self, func):
        """ Wrap a function to profile it.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # self.functions.append(func.__code__)
            # logger.info("prof warpper append %s" % func.__code__)
            code = func.__code__
            begin_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.time()
            self.code_map[code].append((1, end_time - begin_time))
            # logger.info("prof warpper code_map append %s" % code)
            # logger.info("prof warpper add time_stamp %f" % (end_time - begin_time))
            # logger.info("prof warpper pop %s" % self.functions[-1])
            # self.functions.pop()
            return result
        return wrapper

    # def enable_by_count(self):
    #     """ Enable the profiler if it hasn't been enabled before.
    #     """
    #     logger.info("prof enable by count")
    #     if self.enable_count == 0:
    #         self.enable()
    #     self.enable_count += 1

    # def disable_by_count(self):
    #     """ Disable the profiler if the number of disable requests matches the
    #     number of enable request.
    #     """
    #     logger.info("prof disable by count")
    #     if self.enable_count > 0:
    #         self.enable_count -= 1
    #         if self.enable_count == 0:
    #             self.disable()

    # def enable(self):
    #     """ TODO: more precise
    #     """
    #     logger.info("prof enable")
    #     self.begin_time = time.time()

    # def disable(self):
    #     logger.info("prof disable")
    #     self.end_time = time.time()
    #     # What is code?

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
        # print pstats.timings
        show_text(pstats.timings, stream=stream)

    def get_stats(self):
        # stats: {(filename, first_lineno, function_name) : (hits, time)}
        stats = {}
        for code in self.code_map:
            key = label(code)
            stats[key] = self.code_map[code] # TODO
        return FuncCallStats(stats)

    def runctx(self, cmd, globals, locals):
        """ Profile a single executable statement in the given namespaces.
        """
        # logger.info("prof runctx")
        #self.enable_by_count()
        #try:
            #print '\trunctx:', cmd
        exec_(cmd, globals, locals)
            #prof.runctx('execfile_(%r, globals())' % (script_file, ), ns, ns)
        #finally:
        #    #self.disable_by_count()
        #return self

def show_func(filename, first_lineno, func_name, timings, stream=None):
    """ Show results for a single function.
    """
    if stream is None:
        stream = sys.stdout

    nhits = 0
    total_time = 0.0
    for hit, time in timings:
        nhits += 1
        total_time += time
    if nhits == 0:
        percall = 0
    else:
        percall = total_time / nhits
    template = '%10s %10.4f %10.4f    %-s:%s(%s)'
    text = template % (nhits, total_time * 1000, percall * 1000, filename, first_lineno, func_name)
    stream.write(text)
    stream.write("\n")


def show_text(stats, stream=None):
    """ Show text for the given timings.
    """
    if stream is None:
        stream = sys.stdout
    stream.write('File: %s\n' % __file__)
    stream.write('Timer unit: %g s\n\n' % 1e-03)
    template = '%10.8s %10.8s %10.8s    %-s:%s(%s)'
    #template = '%10s %10s %10s    %-s:%s(%s)'
    header = template % ('ncalls', 'tottime', 'percall', 'filename', 'lineno', 'function')
    stream.write("\n")
    stream.write(header)
    stream.write("\n")
    # print stats
    for (fn, lineno, name), timings in sorted(stats.items()):
        show_func(fn, lineno, name, stats[fn, lineno, name], stream=stream)
    stream.write("\n")
    stream.write("\n")


def profile(fn):
    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return prof(fn)(*args, **kwargs)
    return decorated


def find_script(script_name):
    """ Find the script.
    If the input is not a file, then $PATH will be searched.
    """
    if os.path.isfile(script_name):
        return script_name
    path = os.getenv('PATH', os.default).split(os.pathsep)
    for dir in path:
        if dir == '':
            continue
        fn = os.path.join(dir, script_name)
        if os.path.isfile(fn):
            return fn

    sys.stderr.write('Could not find script %s\n' % script_name)
    raise SystemExit(1)

# -p program level
# -f function-call level
# -l line-by-line level

prof = FuncCallProfiler()
def main(args=None):
    if args is None:
        args = sys.argv
    usage = "%prog scriptfile [arg] ..."
    parser = optparse.OptionParser(usage=usage, version="%prog 0.1")
    parser.add_option('-f', '--function-call', action='store_true',
            help="Use the function-call profiler")
    parser.add_option('-o', '--outfile', default=None,
            help="Save stats to <outfile>")
    parser.add_option('-v', '--view', action='store_true',
            help="View the results of the profile in addition to saving it.")

    #if not args[1:]:
    #    parser.print_usage()
    #    sys.exit(2)

    options, option_args = parser.parse_args()

    if not options.outfile:
        extension = 'fprof'
        options.outfile = '%s.%s' % (os.path.basename(option_args[0]), extension)

    if options.function_call:
        prof = FuncCallProfiler()

    script_file = find_script(option_args[0])
    # print 'script:', script_file
    __file__ = script_file

    sys.path.insert(0, os.path.dirname(script_file))

    try:
        try:
            execfile_ = execfile
            ns = locals()
            prof.runctx('execfile_(%r, globals())' % (script_file, ), ns, ns)
        except (KeyboardInterrupt, SystemExit):
            pass
    finally:
        prof.dump_stats(options.outfile)
        print('Wrote profile results to %s' % options.outfile)
        if options.view:
            prof.print_stats()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
