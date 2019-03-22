import tensorflow as tf

from .value import Value
from .. import util
from ..module import Module


# TODO: num_identifiers multiplied by 3


class IdentifierValue(Value):

    def __init__(self, name, capacity=None, embedding_size=None):
        super().__init__(name=name)

        self.capacity = capacity
        if embedding_size is None:
            self.embedding_size = 2 * self.capacity
        else:
            self.embedding_size = embedding_size

    def specification(self):
        spec = super().specification()
        spec.update(embedding_size=self.embedding_size)
        return spec

    def input_size(self):
        return self.embedding_size

    def output_size(self):
        return 0

    def extract(self, data):
        self.num_identifiers = data[self.name].nunique() * 10

    def features(self, x=None):
        features = super().features(x=x)
        if x is None:
            features[self.name] = tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)
        else:
            features[self.name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=(x[self.name],))
            )
        return features

    def tf_initialize(self):
        super().tf_initialize()
        self.placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='input')
        # tf.placeholder_with_default(input=(-1,), shape=(None,), name='input')
        assert self.name not in Module.placeholders
        Module.placeholders[self.name] = self.placeholder
        initializer = util.get_initializer(initializer='normal')
        self.embeddings = tf.get_variable(
            name='embeddings', shape=(self.num_identifiers, self.embedding_size), dtype=tf.float32,
            initializer=initializer, regularizer=None, trainable=False, collections=None,
            caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
            custom_getter=None
        )
        self.current_identifier = tf.get_variable(
            name='current-identifier', shape=(), dtype=tf.int64, trainable=False
        )

    def tf_input_tensor(self, feed=None):
        x = self.placeholder if feed is None else feed[self.name]
        assignment = self.current_identifier.assign(
            value=tf.maximum(x=self.current_identifier, y=tf.reduce_max(input_tensor=x))
        )
        with tf.control_dependencies(control_inputs=(assignment,)):
            x = tf.nn.embedding_lookup(
                params=self.embeddings, ids=x, partition_strategy='mod', validate_indices=True,
                max_norm=None
            )
        return x

    def tf_next_value(self):
        assignment = self.current_identifier.assign_add(delta=1)
        with tf.control_dependencies(control_inputs=(assignment,)):
            return self.current_identifier + 0  # trivial operation to enforce dependency

    def tf_random_value(self, n):
        identifier = tf.random_uniform(
            shape=(n,), minval=0, maxval=self.num_identifiers, dtype=tf.int32, seed=None
        )
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=identifier, partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        return identifier, x
