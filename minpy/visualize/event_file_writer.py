"""Writes events to a disk in a logdir."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import time
import socket
import logging
import six

import minpy.visualization.event_pb2 as event_pb2

import tensorflow as tf

def as_bytes(bytes_or_text, encoding='utf-8'):
    """Converts either bytes or unicode to `bytes`, using utf-8 encoding for text.
    Args:
        bytes_or_text: A `bytes`, `str`, or `unicode` object.
        encoding: A string indicating the charset for encoding unicode.
    Returns:
        A `bytes` object.
    Raises:
        TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, six.text_type):
        return bytes_or_text.encode(encoding)
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text
    else:
        raise TypeError('Expected binary or unicode string, got %r' %
                        (bytes_or_text,))


def directorycheck(path):
    # If the direcotry does not exist, create it!
    if not os.path.exists(path):
        os.makedirs(path)
    # Else, empty the directory.
    else:
        fileList = os.listdir(path)
        for file in fileList:
            os.remove(path + "/" + file)


class EventsWriter:
    """
    @@__init__

    @@WriteEvent

    @@_WriteSerializedEvent
    """

    def __init__(self, file_prefix):
        '''Events files have a name of the form
        '/some/file/path/events.out.tfevents.[timestamp].[hostname]'
        '''
        self._file_prefix = file_prefix + "out.tfevents." \
                            + str(time.time())[:10] + "." + socket.gethostname()

        # Open(Create) the log file with the particular form of name.
        logging.basicConfig(filename=self._file_prefix)

        self._num_outstanding_events = 0

        self._recordio_writer = tf.python_io.TFRecordWriter(self._file_prefix)

        # Initialize an event instance.
        self._event = event_pb2.Event()

        # In events_writer.cc, event.set_wall_time(time_in_seconds) was
        # implemented, which I did not mangage to figure out where it was
        # defined. I assumed this directly modify event.wall_time.
        self._event.wall_time = time.time()

        # Similar as above, event.set_file_version is assumed to be modifying
        # the file_version of an event instance. As asserted in event.proto
        # and events_writer.h, current version is "brain.Event:2" and all
        # versions start with "brain.Event:".
        self._event.file_version = "brain.Event:2"
        self.WriteEvent(self._event)

    def WriteEvent(self, event):
        # Append "event" to the file.

        # Check if event is of type event_pb2.Event proto.
        if not isinstance(event, event_pb2.Event):
            raise TypeError("Expected an event_pb2.Event proto, "
                            " but got %s" % type(event))
        return self._WriteSerializedEvent(event.SerializeToString())

    def _WriteSerializedEvent(self, event_str):
        self._num_outstanding_events += 1
        self._recordio_writer.write(event_str)

    def Flush(self):
        num_outstanding_events_ = 0
        return True

    def Close(self):
        return_value = self.Flush()
        return return_value

class EventFileWriter(object):
    '''Writes `Event` protocol buffers to an event file.

    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format.

    @@__init__

    @@add_event
    @@flush
    @@close
    '''

    def __init__(self, logdir, max_queue=10, flush_secs=120):
        """Creates a `EventFileWriter` and an event file to write to.
        On construction the summary writer creates a new event in `logdir`.
        This event file will contain `Event` protocol buffers, which are written to
        disk via the add_event method.

        The other arguments to the constructor control the asynchronous writes to
        the event file:

        * `flush_seccs`: How often, in seconds, to flush the added summaries
            and events to disk.
        * `max_queue`: Maximum number of summaries or events pending to be
            written to disk before one of the 'add' calls block.

        Args:
            logdir: A string. Directory where event file will be written.
            max_queue: Integer. Size of the queue for pending events and summaries.
            flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk.
        """
        self._logdir = logdir
        directorycheck(self._logdir)
        self._event_queue = six.moves.queue.Queue(max_queue)
        self._ev_writer = EventsWriter(as_bytes(os.path.join(self._logdir, "events")))
        self._closed = False
        self._worker = _EventLoggerThread(self._event_queue, self._ev_writer, flush_secs)

        self._worker.start()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def reopen(self):
        """Reopens the EventFileWriter.

        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.

        Does nothing if the EventFileWriter was not closed.
        """
        if self._closed:
            self._closed = False

    def add_event(self, event):
        """Adds an event to the event file.

        Args:
            event: An `Event` protocol buffer.
        """
        if not self._closed:
            self._event_queue.put(event)

    def flush(self):
        """Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to
        disk.
        """
        self._event_queue.join()
        self._ev_writer.Flush()

    def close(self):
        """Flushes the event file to disk and close the file.

        Call this method when you do not need the summary writer anymore.
        """
        self.flush()
        self._ev_writer.Close()
        self._closed = True

class _EventLoggerThread(threading.Thread):
    """Thread that logs events.

    @@__init__

    @@run
    """

    def __init__(self, queue, ev_writer, flush_secs):
        """Creates an _EventLoggerThread.

        Args:
            queue: A queue from which to dequeue events.
            ev_writer: An event writer. Used to log brain events for
             the visualizer.
            flush_secs: How often, in seconds, to flush the
             pending file to disk.
        """
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._ev_writer = ev_writer
        self._flush_secs = flush_secs
        # The first event will be flushed immediately.
        self._next_event_flush_time = 0

    def run(self):
        while True:
            event = self._queue.get()
            try:
                self._ev_writer.WriteEvent(event)
                # Flush the event writer every so often.
                now = time.time()
                if now > self._next_event_flush_time:
                    self._ev_writer.Flush()
                    # Do it again in 2 minutes.
                    self._next_event_flush_time = now + self._flush_secs
            finally:
                self._queue.task_done()
