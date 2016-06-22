#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# USAGE: - set MINPROF_FLAG to enable the profiler
#        - add @minprof before the function you want to profiling
#        - use minprof(<func>) to wrap a function
#        - use with(<some info>): ... to wrap a code snippet

# TODO: nest, recursive, timeunit, sort
# TODO: GPU

MINPROF_FLAG = True

try:
    import cPickle as pickle
except ImportError:
    import pickle

import functools
import optparse
import time
import os
import sys
import logging

from . import log
logger = log.get_logger(__name__, logging.INFO)

from inspect import getframeinfo, stack, getsource, getfile

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
        # self.unit = unit


def read_lines(filename, begin_lineno, end_lineno):
    # print('read_Lines: %s, %s, %s' % (filename, begin_lineno, end_lineno))
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i >= begin_lineno and i < end_lineno:
                lines.append(line.rstrip())

    return lines


class FuncCallProfiler():
    class Timer():
        def __init__(self, profiler, info=None):
            self.profiler = profiler
            self.info = info
            self.filename = None
            self.begin_lineno = None
            self.end_lineno = None
            self.code_snippet = None

        def __enter__(self):
            # print 'timer enter'
            x = getframeinfo(stack()[1][0])
            self.filename = x.filename
            self.begin_lineno = x.lineno
            self.begin_time = time.time()

        def __exit__(self, type, value, traceback):
            # print 'timer exit'
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
            #print(code)
            if code not in self.profiler.code_map:
                self.profiler.code_map[code] = []
            timing = (1, self.begin_time, self.end_time)
            self.profiler.code_map[code].append(timing)

    def __init__(self, *funcs):
        # logger.info("prof init")
        # self.functions = []
        self.code_map = {}
        self.enable_count = 0
        for func in funcs:
            self.add_function(func)

    def __call__(self, func=None):
        # logger.info("prof __call__")
        # decorator don't use info
        if func is None:
            return FuncCallProfiler.Timer(self)
        if isinstance(func, str):
            return FuncCallProfiler.Timer(self, info=func)
        self.add_function(func)
        wrapper = self.wrap_function(func)
        return wrapper

    def extract_code(self, func, stackdepth):
        try:
            code = func.__code__
        except AttributeError:
            frameinfo = getframeinfo(stack()[stackdepth][0])
            #print(frameinfo)
            #print(func.__name__)
            code = (frameinfo.filename, frameinfo.lineno, func.__name__)
        return code

    def add_function(self, func):
        """ Record function-call profiling information for the
        given Python function.
        """
        #logger.info("prof add function: %s" % func.__code__)
        code = self.extract_code(func, 3)
        if code not in self.code_map:
            self.code_map[code] = []

    def wrap_function(self, func):
        """ Wrap a function to profile it.
        """

        #@functools.wraps(func)
        def wrapper(*args, **kwargs):
            #self.functions.append(func.__code__)
            #logger.info("prof warpper call %s" % func.__code__)
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
        # print pstats.timings
        show_text(pstats.timings, stream=stream)

    def get_stats(self):
        # stats: {(filename, first_lineno, function_name) : (hits, time)}
        stats = {}
        for code in self.code_map:
            key = label(code)
            stats[key] = self.code_map[code]  # TODO
        return FuncCallStats(stats)

    def runctx(self, cmd, globals, locals):
        """ Profile a single executable statement in the given namespaces.
        """
        # logger.info("prof runctx")
        exec_(cmd, globals, locals)


def show_func(filename, first_lineno, func_name, timings, stream=None):
    """ Show results for a single function.
    """
    if stream is None:
        stream = sys.stdout

    nhits = 0
    total_time = 0.0
    for hit, begin_time, end_time in timings:
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
        #template = '%10s %10s %10s    %-s:%s(%s)'
        header = template % ('ncalls', 'tottime', 'percall', 'filename',
                             'lineno', 'function', 'info')
        stream.write("\n")
        stream.write(header)
        stream.write("\n")
        # print stats
        for (fn, lineno, name), timings in sorted(stats.items()):
            show_func(fn, lineno, name, stats[fn, lineno, name], stream=stream)
        stream.write("\n")
        stream.write("\n")


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

minprof = FuncCallProfiler()


def main(args=None):
    # print '%s' % globals()
    if args is None:
        args = sys.argv
    usage = "%prog scriptfile [arg] ..."
    parser = optparse.OptionParser(usage=usage, version="%prog 0.1")
    parser.add_option('-f',
                      '--function-call',
                      action='store_true',
                      help="Use the function-call profiler")
    parser.add_option('-o',
                      '--outfile',
                      default=None,
                      help="Save stats to <outfile>")
    parser.add_option(
        '-v',
        '--view',
        action='store_true',
        help="View the results of the profile in addition to saving it.")

    #if not args[1:]:
    #    parser.print_usage()
    #    sys.exit(2)

    options, option_args = parser.parse_args()

    if not options.outfile:
        extension = 'fprof'
        options.outfile = '%s.%s' % (os.path.basename(option_args[0]),
                                     extension)

    # if options.function_call:
    #     prof = FuncCallProfiler()

    script_file = find_script(option_args[0])
    # print 'script:', script_file
    __file__ = script_file

    sys.path.insert(0, os.path.dirname(script_file))

    try:
        try:
            execfile_ = execfile
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
