import os
import shutil
import time
from functools import wraps
from typing import Dict, Optional, List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.summary_ops_v2 import SummaryWriter

from .util import Profiler, ProfilerArgs


class Module(object):
    def __init__(self, name: str, summarizer_dir: str = None, profiler_args: ProfilerArgs = None):
        self.name = name
        self.summarizer_dir = summarizer_dir
        self.summarizer: Optional[SummaryWriter] = None
        self.profiler_args = profiler_args
        self.profiler: Optional[Profiler] = None
        self.submodules: List[Module] = list()
        self.initialized: bool = False
        self.global_step: Optional[tf.Tensor] = None

    def specification(self):
        return dict(name=self.name)

    def __repr__(self):
        return '{name}({spec})'.format(
            name=self.__class__.__name__,
            spec=','.join(
                '{key}={value}'.format(key=key, value=repr(value))
                for key, value in self.specification().items()
            )
        )

    def __str__(self):
        return repr(self)

    def module_initialize(self):
        pass

    def initialize(self):
        if self.initialized:
            raise NotImplementedError
        self.initialized = True

        with tf.compat.v1.variable_scope(name_or_scope=self.make_tf_compatible(string=self.name)):
            for submodule in self.submodules:
                submodule.initialize()
            self.module_initialize()

    def add_module(self, module, **kwargs):
        if isinstance(module, dict):
            for key, value in module.items():
                if kwargs.get(key, value) != value:
                    raise ValueError
                kwargs[key] = value
            module = kwargs.pop('module')
            return self.add_module(module=module, **kwargs)
        elif isinstance(module, str):
            assert module in module_registry, module
            module = module_registry[module]
            return self.add_module(module=module, **kwargs)
        elif issubclass(module, Module):
            module = module(**kwargs)
            self.submodules.append(module)
            return module
        else:
            raise NotImplementedError

    def __enter__(self):
        self.graph = tf.Graph()
        self.global_step = tf.compat.v1.train.get_or_create_global_step(graph=self.graph)

        with self.graph.as_default():
            if self.profiler_args is not None:
                self.profiler = Profiler(filepath=self.profiler_args.filepath, period=self.profiler_args.period)

            if self.summarizer_dir is not None:
                directories = sorted(os.listdir(self.summarizer_dir))
                if len(directories) > 6:
                    for subdir in directories[:-6]:
                        subdir = os.path.join(self.summarizer_dir, subdir)
                        subdir = os.path.abspath(subdir)
                        if os.path.isdir(subdir):
                            shutil.rmtree(subdir)
                        else:
                            os.remove(subdir)
                with tf.name_scope(name='summarizer'):
                    self.summarizer = tf.contrib.summary.create_file_writer(
                        logdir=os.path.join(self.summarizer_dir, time.strftime("%Y%m%d-%H%M%S")),
                        max_queue=None, flush_millis=10000, filename_suffix=None
                    )

                # tf.contrib.summary.record_summaries_every_n_global_steps(n=100, global_step=None)
                with self.summarizer.as_default(), tf.contrib.summary.always_record_summaries():
                    self.initialize()

                    with tf.name_scope(name='initialization', default_name=None, values=None):
                        summarizer_init = tf.contrib.summary.summary_writer_initializer_op()
                        assert len(summarizer_init) == 1
                        initialization = (tf.compat.v1.global_variables_initializer(), summarizer_init[0])
                        self.summarizer_close = self.summarizer.close()
                        graph_def = self.graph.as_graph_def(from_version=None, add_shapes=True)
                        graph_str = tf.constant(
                            value=graph_def.SerializeToString(), dtype=tf.string, shape=(),
                            verify_shape=False
                        )
                        graph_summary = tf.contrib.summary.graph(
                            param=graph_str, step=self.global_step
                        )

            else:
                self.initialize()
                initialization = tf.compat.v1.global_variables_initializer()

        self.graph.finalize()
        self.session = tf.compat.v1.Session(target='', graph=self.graph, config=None)
        self.session.__enter__()
        self.run(fetches=initialization)
        if self.summarizer is not None:
            self.run(fetches=graph_summary)
        return self

    def run(self, fetches: tf.Tensor, feed_dict: Dict[tf.Tensor, np.ndarray] = None):
        options, run_metadata = None, None
        if self.profiler is not None:
            if self.profiler.is_trace_step():
                options, run_metadata = self.profiler.get_options_and_metadata()

        result = self.session.run(
                fetches=fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata
            )

        if self.profiler is not None:
            if self.profiler.is_trace_step():
                self.profiler.read_trace(run_metadata)
            self.profiler.increment()
        return result

    def __exit__(self, type, value, traceback):
        if self.summarizer is not None:
            self.run(fetches=self.summarizer_close)
        self.session.__exit__(type, value, traceback)
        tf.compat.v1.reset_default_graph()
        if self.profiler is not None:
            self.profiler.write_traces()

    def make_tf_compatible(self, string):
        # TODO: make it always match [A-Za-z0-9_.\\-/]*
        return string.replace(' ', '_').replace(':', '').replace('%', '').strip('_')


module_registry: Dict[str, Module] = dict()


def register(name, module):
    assert name not in module_registry
    module_registry[name] = module


def tensorflow_name_scoped(tf_function):
    @wraps(tf_function)
    def function(self, *args, **kwargs):
        name = "{}.{}".format(self.name, tf_function.__name__)
        with tf.name_scope(name=name.replace(' ', '_').replace(':', '').replace('%', '')):
            results = tf_function(self, *args, **kwargs)
        return results

    return function
