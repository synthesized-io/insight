import json
import tensorflow as tf
from tensorflow.python.client import timeline
from recordclass import RecordClass
from collections import namedtuple


ProfilerArgs = namedtuple("ProfilerArgs", "filepath period")


class Profiler(RecordClass):
    filepath: str = ""
    period: int = 1
    step: int = 0
    traces: list = []

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
        return tf.ones_initializer(dtype=tf.float32)
    elif initializer == 'zeros':
        return tf.zeros_initializer(dtype=tf.float32)
    elif initializer == 'zeros-int':
        return tf.zeros_initializer(dtype=tf.int64)
    else:
        raise NotImplementedError


def get_regularizer(regularizer, weight):
    assert weight >= 0.0
    if regularizer == 'none' or weight == 0.0:
        return tf.no_regularizer
    elif regularizer == 'l2':
        return tf.contrib.layers.l2_regularizer(scale=weight, scope=None)
    else:
        raise NotImplementedError
