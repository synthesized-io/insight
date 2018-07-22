import tensorflow as tf
from synthesized.core import util
from synthesized.core.values import Value


class IdentifierValue(Value):

    def __init__(self, name, num_identifiers, embedding_size):
        super().__init__(name=name)
        self.num_identifiers = num_identifiers
        self.embedding_size = embedding_size

    def specification(self):
        spec = super().specification()
        spec.update(num_identifiers=self.num_identifiers, embedding_size=self.embedding_size)
        return spec

    def input_size(self):
        return self.embedding_size

    def output_size(self):
        return 0

    def feature(self, x=None):
        if x is None:
            return tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=(x,)))

    def tf_initialize(self):
        super().tf_initialize()
        self.placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='input')
        # tf.TensorArray???
        # self.identifiers = tf.get_variable(
        #     name='identifiers', shape=(self.num_identifiers,), dtype=tf.int32,
        #     initializer=util.initializers['zero_int'], regularizer=None, trainable=False,
        #     collections=None, caching_device=None, partitioner=None, validate_shape=True,
        #     use_resource=None, custom_getter=None
        # )
        self.embeddings = tf.get_variable(
            name='embeddings', shape=(self.num_identifiers, self.embedding_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=None, trainable=False,
            collections=None, caching_device=None, partitioner=None, validate_shape=True,
            use_resource=None, custom_getter=None
        )

    def tf_input_tensor(self, feed=None):
        # max_index = len(self.embeddings)???
        # new_max_index = tf.reduce_max(input_tensor=self.placeholder, axis=1, keepdims=False)
        # if max_index < new_max_index:
        #     tf.random_normal(
        #         shape=(new_max_index - max_index, self.embedding_size), mean=0.0, stddev=1.0,
        #         dtype=tf.float32, seed=None
        #     )
        x = self.placeholder if feed is None else feed
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=x, partition_strategy='mod', validate_indices=True,
            max_norm=None
        )
        return x

    def random_value(self, n):
        identifier = tf.random_uniform(
            shape=(n,), minval=0, maxval=self.num_identifiers, dtype=tf.int32, seed=None
        )
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=identifier, partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        return identifier, x
