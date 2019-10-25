import json
import re

import tensorflow as tf
from tensorflow.python.client import timeline
from collections import namedtuple

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
        trace_events = json.loads(chrome_trace)
        trace_events["step"] = self.step
        self.traces.append(trace_events)

    def write_traces(self):
        with open(self.filepath, 'w') as out_file:
            json.dump(self.traces, out_file)


def get_initializer(initializer):
    if initializer == 'normal':
        return tf.random_normal_initializer(mean=0.0, stddev=1e-2)
    elif initializer == 'normal-large':
        return tf.random_normal_initializer(mean=0.0, stddev=1.0)
    elif initializer == 'orthogonal':
        return tf.orthogonal_initializer(gain=1.0)
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


def compute_embedding_size(num_categories: int, capacity: int) -> int:
    if capacity is None:
        capacity = 128
    if num_categories <= 10:
        return num_categories
    else:
        return min(int(num_categories * 0.75), capacity)
