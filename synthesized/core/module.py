import tensorflow as tf


class Module(object):

    placeholders = None

    def __init__(self, name, submodules=()):
        self.name = name
        self.submodules = list(submodules)
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

    def tf_initialize(self):
        pass

    @staticmethod
    def create_tf_function(tf_function, name):

        def function(*args, **kwargs):
            with tf.name_scope(name=name.replace(' ', '_').replace(':', ''), default_name=None, values=None):
                results = tf_function(*args, **kwargs)
            return results

        return function

    def initialize(self):
        if self.initialized:
            raise NotImplementedError
        self.initialized = True

        # TODO: hp_ for hyperparameter, for automatic collection and placeholder?
        for function_name in dir(self):
            if not function_name.startswith('tf_') or function_name == 'tf_initialize':
                continue
            if hasattr(self, function_name[3:]):
                raise NotImplementedError
            tf_function = getattr(self, function_name)
            if not callable(tf_function):
                raise NotImplementedError
            fct_name = '{name}.{function}'.format(name=self.name, function=function_name[3:])
            function = Module.create_tf_function(tf_function=tf_function, name=fct_name)
            setattr(self, function_name[3:], function)

        with tf.variable_scope(
            name_or_scope=self.name.replace(' ', '_').replace(':', ''), default_name=None, values=None,
            initializer=None, regularizer=None, caching_device=None, partitioner=None,
            custom_getter=None, reuse=None, dtype=None, use_resource=None, constraint=None,
            auxiliary_name_scope=True
        ):
            for submodule in self.submodules:
                submodule.initialize()
            self.tf_initialize()

    def add_module(self, module, modules=None, **kwargs):
        if isinstance(module, dict):
            for key, value in module.items():
                if kwargs.get(key, value) != value:
                    raise ValueError
                kwargs[key] = value
            module = module.pop('module')
            return self.add_module(module=module, modules=modules, **kwargs)
        elif isinstance(module, str):
            if modules is None or module not in modules:
                raise NotImplementedError
            module = modules[module]
            return self.add_module(module=module, modules=None, **kwargs)
        elif issubclass(module, Module):
            module = module(**kwargs)
            self.submodules.append(module)
            return module
        else:
            raise NotImplementedError

    def __enter__(self):
        assert Module.placeholders is None
        Module.placeholders = dict()
        self.graph = tf.Graph()
        self.global_step = tf.train.get_or_create_global_step(graph=self.graph)
        with self.graph.as_default():
            self.initialize()
            initialize = tf.global_variables_initializer()
            # self.summary_writer = tf.contrib.summary.create_file_writer(
            #     logdir='logs', max_queue=None, flush_millis=None, filename_suffix=None, name=None
            # )
            # with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            #     self.initialize()
            #     initialize = (tf.global_variables_initializer(), self.summary_writer.init())
                # graph_def = self.graph.as_graph_def(from_version=None, add_shapes=True)
                # graph_str = tf.constant(
                #     value=graph_def.SerializeToString(), dtype=tf.string, shape=(), name=None,
                #     verify_shape=False
                # )
                # graph_summary = tf.contrib.summary.graph(
                #     param=graph_str, step=self.global_step, name=None
                # )
        self.graph.finalize()
        self.placeholders = Module.placeholders
        Module.placeholders = None
        # self.summary_writer = tf.summary.FileWriter(
        #     logdir='logs', graph=self.graph, max_queue=10, flush_secs=120, filename_suffix=None
        # )
        self.session = tf.Session(target='', graph=self.graph, config=None)
        self.session.__enter__()
        self.run(fetches=initialize)  # , feed_dict={name: () for name in self.placeholders})
        # self.session.run(fetches=graph_summary, feed_dict=None, options=None, run_metadata=None)
        # with self.summary_writer.as_default():
        #     tf.contrib.summary.initialize(graph=self.graph, session=self.session)
        return self

    def run(self, fetches, feed_dict=None):
        if feed_dict is not None:
            feed_dict = {self.placeholders[name]: value for name, value in feed_dict.items()}
        return self.session.run(
            fetches=fetches, feed_dict=feed_dict, options=None, run_metadata=None
        )

    def __exit__(self, type, value, traceback):
        self.session.__exit__(type, value, traceback)
