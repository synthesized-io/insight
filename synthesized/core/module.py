import tensorflow as tf


class Module(object):

    def __init__(self, name, submodules=(), master=False):
        super().__init__()
        self.name = name
        self.submodules = list(submodules)
        self.master = master
        self.initialized = False

    def initialize(self):
        if self.initialized:
            raise NotImplementedError
        self.initialized = True
        if self.master:
            tf.reset_default_graph()
        with tf.variable_scope(
            name_or_scope=self.name, default_name=None, values=None,
            initializer=None, regularizer=None, caching_device=None, partitioner=None,
            custom_getter=None, reuse=None, dtype=None, use_resource=None, constraint=None,
            auxiliary_name_scope=True
        ):
            for submodule in self.submodules:
                submodule.initialize()
            self._initialize()
        if self.master:
            initialize = tf.global_variables_initializer()
            tf.get_default_graph().finalize()
            self.session = tf.Session(target='', graph=None, config=None)
            self.session.__enter__()
            self.session.run(fetches=initialize, feed_dict=None, options=None, run_metadata=None)

    def _initialize(self):
        raise NotImplementedError
