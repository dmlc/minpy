from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import bisect
import minpy
import random

from minpy.visualization.summary_pb2 import Summary, HistogramProto

def scalarSummary(TAG, scalar):
    if isinstance(scalar, minpy.array.Array) == True:
      scalar = float(scalar.asnumpy()[0])
    elif isinstance(scalar, numpy.ndarray) == True:
      scalar = float(scalar[0])
    return Summary(value=[Summary.Value(tag=TAG, simple_value=scalar)])

def histogramSummary(TAG, values):
    histogram = _MakeHistogram(values.asnumpy().astype(float))
    return Summary(value=[Summary.Value(tag=TAG, histo=histogram)])

def _MakeHistogramBuckets():
    v = 1E-12
    buckets = []
    neg_buckets = []
    while v < 1E20:
        buckets.append(v)
        neg_buckets.append(-v)
        v *= 1.1
    # Should include DBL_MAX, but won't bother for test data.
    return neg_buckets[::-1] + [0] + buckets

def _MakeHistogram(values):
    """Convert values into a histogram proto using logic from histogram.cc."""
    limits = _MakeHistogramBuckets()
    counts = [0] * len(limits)
    for v in values:
      idx = bisect.bisect_left(limits, v)
      counts[idx] += 1

    limit_counts = [(limits[i], counts[i]) for i in xrange(len(limits))
                   if counts[i]]
    bucket_limit = [lc[0] for lc in limit_counts]
    bucket = [lc[1] for lc in limit_counts]
    sum_sq = sum(v * v for v in values)
    return HistogramProto(min=min(values),
                          max=max(values),
                          num=len(values),
                          sum=sum(values),
                          sum_squares=sum_sq,
                          bucket_limit=bucket_limit,
                          bucket=bucket)

def WriteHistogramSeries(writer, tag, mu_sigma_tuples, n=20):
  """Write a sequence of normally distributed histograms to writer."""
  step = 0
  wall_time = _start_time
  for [mean, stddev] in mu_sigma_tuples:
    data = [random.normalvariate(mean, stddev) for _ in xrange(n)]
    histo = _MakeHistogram(data)
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=histo)])
    event = tf.Event(wall_time=wall_time, step=step, summary=summary)
    writer.add_event(event)
    step += 10
    wall_time += 100
