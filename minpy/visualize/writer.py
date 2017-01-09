'''It is for creating and writing summaries.
Based on the code from tensorflow/tensorflow/python/summary/writer/writer.py.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import minpy.visualize.summary_pb2 as summary_pb2
import minpy.visualize.event_pb2 as event_pb2
from minpy.visualize.event_file_writer import EventFileWriter

class SummaryWriter(object):
    '''Creates a `SummaryWriter` and an event file.'''
    def __init__(self, logdir, max_queue=10, flush_secs=120):
        """On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, `add_session_log()`,
        or `add_event()`.

        Args:
        - logdir: A string. Directory where event file will be written.
        - max_queue: Integer. Size of the queue for pending events and summaries.
        - flush_secs: Number. How often, in seconds, to flush the pending events
        and summaries to disk.
        """
        self.logdir = logdir
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.event_writer = EventFileWriter(self.logdir, self.max_queue,
                                            self.flush_secs)

        # For storing used tags for session.run() outputs.
        self._session_run_tags = {}

    def add_summary(self, summary, global_step=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.

        You can pass a `summary_pb2.Summary` protocol buffer that you populate
        with your own data.

        Args:
        - summary: A `Summary` protocol buffer, optionally serialized as a string.
        - global_step: Number. Optional global step value to record with the
                   summary.
        """
        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step)

    def add_session_log(self, session_log, global_step=None):
        """Adds a `SessionLog` protocol buffer to the event file.
        This method wraps the provided session in an `Event` protocol buffer
        and adds it to the event file.

        Args:
        - session_log: A `SessionLog` protocol buffer.
        - global_step: Number. Optional global step value to record with the
                       summary.
        """
        event = event_pb2.Event(session_log=session_log)
        self.add_event(event, global_step)

    def add_run_metadata(self, run_metadata, tag, global_step=None):
        """Adds a metadata information for a single session.run() call.

        Args:
        - run_metadata: A `RunMetadata` protobuf object.
        - tag: The tag name for this metadata.
        - global_step: Number. Optional global step counter to record with the
                       StepStats.

        Raises:
        - ValueError: If the provided tag was already used for this type of event.
        """
        if tag in self._session_run_tags:
            raise ValueError("The provided tag was already used for this event type")
        self._session_run_tags[tag] = True

        tagged_metadata = event_pb2.TaggedRunMetadata()
        tagged_metadata.tag = tag
        # Store the `RunMetadata` object as bytes in order to have postponed
        # (lazy) deserialization when used later.
        tagged_metadata.run_metadata = run_metadata.SerializeToString()
        event = event_pb2.Event(tagged_run_metadata=tagged_metadata)
        self.add_event(event, global_step)

    def add_event(self, event, step):
        """Calls self.event_writer to write event at corresponding step.

        Args:
        - event: A `event` protobuf object.
        - step: Number.
        """
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)

    def get_logdir(self):
        '''Return the directory for log files.'''
        return self.logdir

    def flush(self):
        '''Calls self.event_writer to flush the events.'''
        self.event_writer.flush()

    def close(self):
        '''Closes the self.event_writer.'''
        self.event_writer.close()

    def reopen(self):
        '''Reopens the self.event_writer.'''
        self.event_writer.reopen()
