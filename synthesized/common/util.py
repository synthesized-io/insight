import simplejson
import re

import tensorflow as tf
from tensorflow.python.client import timeline
from collections import namedtuple
import numpy as np
from pyemd import emd

RE_START = re.compile(r"^[^A-Za-z0-9.]")
RE_END = re.compile(r"[^A-Za-z0-9_.\-/]")

ProfilerArgs = namedtuple("ProfilerArgs", "filepath period")


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
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
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
        return tf.random_normal_initializer(mean=0.0, stddev=1e-2)
    elif initializer == 'normal-small':
        # for embeddings - see https://arxiv.org/pdf/1711.09160.pdf
        return tf.random_normal_initializer(mean=0.0, stddev=1e-3)
    elif initializer == 'normal-large':
        return tf.random_normal_initializer(mean=0.0, stddev=1.0)
    elif initializer == 'glorot-normal':
        return tf.glorot_normal_initializer()
    elif initializer == 'orthogonal':
        return tf.orthogonal_initializer(gain=1.0)
    elif initializer == 'orthogonal-small':
        return tf.orthogonal_initializer(gain=1e-2)
    elif initializer == 'ones':
        return tf.ones_initializer()
    elif initializer == 'zeros':
        return tf.zeros_initializer()

    else:
        raise NotImplementedError


def get_regularizer(regularizer, weight):
    assert weight >= 0.0
    if regularizer == 'none' or weight == 0.0:
        return tf.compat.v1.no_regularizer
    elif regularizer == 'l2':
        return tf.contrib.layers.l2_regularizer(scale=weight, scope=None)
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
