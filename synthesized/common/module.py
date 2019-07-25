import os
import time
from functools import wraps
from typing import Dict, Optional

import tensorflow as tf


class Module(object):
    placeholders: Optional[Dict[str, tf.Tensor]] = None
    global_step: Optional[tf.Tensor] = None
    summarizer = None

    def __init__(self, name, summarizer=None):
        self.name = name
        self._summarizer = summarizer

        self.submodules = list()
        self.initialized = False

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

        with tf.variable_scope(name_or_scope=make_tf_compatible(string=self.name)):
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

    def add_placeholder(self, name, dtype, shape):
        if name in Module.placeholders:
            raise NotImplementedError

        Module.placeholders[name] = tf.placeholder(
            dtype=dtype, shape=shape, name=make_tf_compatible(string=name)
        )

    def __enter__(self):
        Module.placeholders = dict()
        self.graph = tf.Graph()
        Module.global_step = tf.train.get_or_create_global_step(graph=self.graph)
        Module.summarizer = self._summarizer

        with self.graph.as_default():

            if Module.summarizer is not None:
                directories = sorted(os.listdir(Module.summarizer))
                if len(directories) > 6:
                    for subdir in directories[:-6]:
                        subdir = os.path.join(Module.summarizer, subdir)
                        os.remove(os.path.join(subdir, os.listdir(subdir)[0]))
                        os.rmdir(subdir)
                with tf.name_scope(name='summarizer'):
                    Module.summarizer = tf.contrib.summary.create_file_writer(
                        logdir=os.path.join(Module.summarizer, time.strftime("%Y%m%d-%H%M%S")),
                        max_queue=None, flush_millis=10000, filename_suffix=None
                    )

                # tf.contrib.summary.record_summaries_every_n_global_steps(n=100, global_step=None)
                with Module.summarizer.as_default(), tf.contrib.summary.always_record_summaries():
                    self.initialize()

                    with tf.name_scope(name='initialization', default_name=None, values=None):
                        summarizer_init = tf.contrib.summary.summary_writer_initializer_op()
                        assert len(summarizer_init) == 1
                        initialization = (tf.global_variables_initializer(), summarizer_init[0])
                        self.summarizer_close = Module.summarizer.close()
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
                initialization = tf.global_variables_initializer()

        self.graph.finalize()
        self.session = tf.Session(target='', graph=self.graph, config=None)
        self.session.__enter__()
        self.run(fetches=initialization)
        if Module.summarizer is not None:
            self.run(fetches=graph_summary)
        return self

    def run(self, fetches, feed_dict=None):
        if feed_dict is not None:
            feed_dict = {Module.placeholders[name]: value for name, value in feed_dict.items()}
        return self.session.run(fetches=fetches, feed_dict=feed_dict)

    def __exit__(self, type, value, traceback):
        if Module.summarizer is not None:
            self.run(fetches=self.summarizer_close)
        self.session.__exit__(type, value, traceback)
        tf.reset_default_graph()
        Module.placeholders = None


module_registry: Dict[str, Module] = dict()


def register(name, module):
    assert name not in module_registry
    module_registry[name] = module


def make_tf_compatible(string):
    return string.replace(' ', '_').replace(':', '').replace('%', '')


def tensorflow_name_scoped(tf_function):
    @wraps(tf_function)
    def function(self, *args, **kwargs):
        name = "{}.{}".format(self.name, tf_function.__name__)
        with tf.name_scope(name=name.replace(' ', '_').replace(':', '').replace('%', '')):
            results = tf_function(self, *args, **kwargs)
        return results

    return function