import tensorflow as tf
from functools import wraps


def tensorflow_name_scoped(tf_function):
    @wraps(tf_function)
    def function(self, *args, **kwargs):
        name = "{}.{}".format(self.name, tf_function.__name__)
        with tf.name_scope(name=name.replace(' ', '_').replace(':', '').replace('%', '')):
            results = tf_function(self, *args, **kwargs)
        return results

    return function


class Module(object):
    placeholders = None
    global_step = None

    def __init__(self, name, summarizer=False):
        self.name = name
        self.summarizer = summarizer

        self.submodules = list()
        self.placeholders = None
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

        with tf.variable_scope(name_or_scope=self.name.replace(' ', '_').replace(':', '').replace('%', '')):
            for submodule in self.submodules:
                submodule.initialize()
            self.module_initialize()

    def add_module(self, module, modules=None, **kwargs):
        if isinstance(module, dict):
            for key, value in module.items():
                if kwargs.get(key, value) != value:
                    raise ValueError
                kwargs[key] = value
            module = kwargs.pop('module')
            return self.add_module(module=module, modules=modules, **kwargs)
        elif isinstance(module, str):
            assert modules is not None and module in modules, module
            module = modules[module]
            return self.add_module(module=module, modules=None, **kwargs)
        elif issubclass(module, Module):
            module = module(**kwargs)
            self.submodules.append(module)
            return module
        else:
            raise NotImplementedError

    # def add_summary(self, name, tensor):
    #     # name = '{}-{}'.format(self.name, name)
    #     shape = tuple(tensor.get_shape().as_list())
    #     if shape == () or shape == (-1):
    #         summary = tf.contrib.summary.scalar(name=name, tensor=tensor, family=None, step=None)

    #     elif shape == (1,) or shape == (-1, 1):
    #         tensor = tf.squeeze(input=tensor, axis=-1)
    #         summary = tf.contrib.summary.scalar(name=name, tensor=tensor, family=None, step=None)

    #     else:
    #         summary = tf.contrib.summary.histogram(name=name, tensor=tensor, family=None, step=None)

    #     self.summaries[name] = summary

    def __enter__(self):
        print('hi')
        Module.placeholders = dict()
        self.graph = tf.Graph()
        Module.global_step = tf.train.get_or_create_global_step(graph=self.graph)
        with self.graph.as_default():

            if self.summarizer:
                with tf.name_scope(name='summarizer', default_name=None, values=None):
                    self.summarizer = tf.contrib.summary.create_file_writer(
                        logdir=('summaries_' + self.name), max_queue=None, flush_millis=10000,
                        filename_suffix=None
                    )

                # tf.contrib.summary.record_summaries_every_n_global_steps(n=100, global_step=None)
                with self.summarizer.as_default(), tf.contrib.summary.always_record_summaries():
                    self.initialize()

                    with tf.name_scope(name='initialization', default_name=None, values=None):
                        summarizer_init = tf.contrib.summary.summary_writer_initializer_op()
                        assert len(summarizer_init) == 1
                        initialization = (tf.global_variables_initializer(), summarizer_init[0])
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
                self.summarizer = None
                self.initialize()
                initialization = tf.global_variables_initializer()

        self.graph.finalize()
        self.placeholders = Module.placeholders
        Module.placeholders = None
        self.session = tf.Session(target='', graph=self.graph, config=None)
        self.session.__enter__()
        self.run(fetches=initialization)
        if self.summarizer is not None:
            self.run(fetches=graph_summary)
        return self

    def run(self, fetches, feed_dict=None):
        if feed_dict is not None:
            feed_dict = {self.placeholders[name]: value for name, value in feed_dict.items()}
        return self.session.run(
            fetches=fetches, feed_dict=feed_dict, options=None, run_metadata=None
        )

    def __exit__(self, type, value, traceback):
        if self.summarizer is not None:
            self.run(fetches=self.summarizer_close)
        self.session.__exit__(type, value, traceback)
        Module.placeholders = None
