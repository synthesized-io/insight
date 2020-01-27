from functools import wraps
from typing import Dict

import tensorflow as tf

from .util import make_tf_compatible

module_registry: Dict[str, tf.Module] = dict()


def register(name, module):
    assert name not in module_registry
    module_registry[name] = module


def tensorflow_name_scoped(tf_function):
    @wraps(tf_function)
    def function(self, *args, **kwargs):
        name = "{}.{}".format(self._name, tf_function.__name__)
        with tf.name_scope(name=make_tf_compatible(name)):
            results = tf_function(self, *args, **kwargs)
        return results

    return function
