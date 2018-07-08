import tensorflow as tf


class Module(object):

    def __init__(self, name, submodules=(), master=False):
        self.name = name
        self.submodules = list(submodules)
        self.master = master
        self.initialized = False

    def _initialize(self):
        pass

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
        if not self.master:
            raise NotImplementedError
        self.initialize()
        initialize = tf.global_variables_initializer()
        tf.get_default_graph().finalize()
        self.session = tf.Session(target='', graph=None, config=None)
        self.session.__enter__()
        self.session.run(fetches=initialize, feed_dict=None, options=None, run_metadata=None)
        return self

    def __exit__(self, type, value, traceback):
        if not self.master:
            raise NotImplementedError
        self.session.__exit__(type, value, traceback)
