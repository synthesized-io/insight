import simplejson
import re
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.eager import context
import numpy as np
from pyemd import emd

RE_START = re.compile(r"^[^A-Za-z0-9.]")
RE_END = re.compile(r"[^A-Za-z0-9_./]")

ProfilerArgs = namedtuple("ProfilerArgs", "filepath period")


def record_summaries_every_n_global_steps(n: int, global_step: tf.Variable):
    """Sets the should_record_summaries Tensor to true if global_step % n == 0."""
    with tf.device("cpu:0"):
        if n != 0:
            def should():
                return tf.math.equal(global_step % tf.constant(n, dtype=tf.int64), 0)

            if not context.executing_eagerly():
                should = should()
        else:
            should = tf.constant(False, dtype=tf.bool)

    return tf.summary.record_if(should)


class Profiler:
    def __init__(self, filepath: str = "", period: int = 1, step: int = 0, traces: list = list()):
        self.filepath = filepath
        self.period = period
        self.step = step
        self.traces = traces

    def is_trace_step(self):
        return self.step % self.period == 0

    def increment(self):
        self.step += 1

    @staticmethod
    def get_options_and_metadata():
        options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
        return options, run_metadata

    def read_trace(self, run_metadata):
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        trace_events = simplejson.loads(chrome_trace)
        trace_events["step"] = self.step
        self.traces.append(trace_events)

    def write_traces(self):
        with open(self.filepath, 'w') as out_file:
            simplejson.dump(self.traces, out_file, ignore_nan=True)


def get_initializer(initializer):
    if initializer == 'normal':
        return tf.initializers.RandomNormal(mean=0.0, stddev=1e-2)
    elif initializer == 'normal-small':
        return tf.initializers.RandomNormal(mean=0.0, stddev=1e-3)
    elif initializer == 'normal-large':
        return tf.initializers.RandomNormal(mean=0.0, stddev=1.0)
    elif initializer == 'glorot-normal':
        return tf.initializers.glorot_normal()
    elif initializer == 'orthogonal':
        return tf.initializers.orthogonal(gain=1.0)
    elif initializer == 'orthogonal-small':
        return tf.initializers.orthogonal(gain=1e-2)
    elif initializer == 'ones':
        return tf.initializers.ones()
    elif initializer == 'zeros':
        return tf.initializers.zeros()

    else:
        raise NotImplementedError


def get_regularizer(regularizer, weight):
    assert weight >= 0.0
    if regularizer == 'none' or weight == 0.0:
        return tf.compat.v1.no_regularizer
    elif regularizer == 'l2':
        return tf.keras.regularizers.l2(0.5 * (weight))
    else:
        raise NotImplementedError


def make_tf_compatible(string):
    return re.sub(RE_END, '_', re.sub(RE_START, '.', str(string)))


def categorical_emd(a, b):
    space = list(set(a).union(set(b)))

    # To protect from memory errors:
    if len(space) >= 1e4:
        return 0.

    a_unique, counts = np.unique(a, return_counts=True)
    a_counts = dict(zip(a_unique, counts))

    b_unique, counts = np.unique(b, return_counts=True)
    b_counts = dict(zip(b_unique, counts))

    p = np.array([float(a_counts[x]) if x in a_counts else 0.0 for x in space])
    q = np.array([float(b_counts[x]) if x in b_counts else 0.0 for x in space])

    p /= np.sum(p)
    q /= np.sum(q)

    distances = 1 - np.eye(len(space))

    return emd(p, q, distances)
