import tensorflow as tf

from .value import Value
from .. import util
from ..module import Module


# TODO: num_identifiers multiplied by 3


class IdentifierValue(Value):

    def __init__(self, name, embedding_size, num_identifiers=None):
        super().__init__(name=name)

        self.embedding_size = embedding_size
        self.num_identifiers = num_identifiers

    def specification(self):
        spec = super().specification()
        spec.update(num_identifiers=self.num_identifiers, embedding_size=self.embedding_size)
        return spec

    def input_size(self):
        return self.embedding_size

    def output_size(self):
        return 0

    def placeholders(self):
        raise NotImplementedError
        yield self.placeholder

    def extract(self, data):
        if self.num_identifiers is None:
            self.num_identifiers = data[self.name].nunique() * 3
        elif data[self.name].nunique() > self.num_identifiers:
            raise NotImplementedError

    def preprocess(self, data):
        # normalization = {x: n for n, x in enumerate(data[self.name].unique())}
        # data[self.name] = data[self.name].map(arg=normalization)
        return data

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
        assert self.name not in Module.placeholders
        Module.placeholders[self.name] = self.placeholder
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
        x = self.placeholder if feed is None else feed[self.name]
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=x, partition_strategy='mod', validate_indices=True,
            max_norm=None
        )
        return x

    def random_value(self, n, multiples=1):
        identifier = tf.random_uniform(
            shape=(n,), minval=0, maxval=self.num_identifiers, dtype=tf.int32, seed=None
        )
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=identifier, partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        identifier = tf.expand_dims(input=identifier, axis=0)
        identifier = tf.tile(input=identifier, multiples=tf.stack(values=(multiples, 1), axis=0))
        identifier = tf.reshape(tensor=identifier, shape=tf.expand_dims(input=(n * multiples), axis=0))
        x = tf.expand_dims(input=x, axis=0)
        x = tf.tile(input=x, multiples=tf.stack(values=(multiples, 1, 1), axis=0))
        x = tf.reshape(tensor=x, shape=tf.stack(values=(n * multiples, self.embedding_size), axis=0))
        return identifier, x
