import tensorflow as tf

from .value import Value
from .. import util
from ..module import Module, tensorflow_name_scoped


# TODO: num_identifiers multiplied by 3


class IdentifierValue(Value):

    def __init__(
        self, name, identifiers=None, capacity=None, embedding_size=None
    ):
        super().__init__(name=name)

        if identifiers is None:
            self.identifiers = None
            self.num_identifiers = None
        elif isinstance(identifiers, int):
            self.identifiers = self.num_identifiers = identifiers
        else:
            self.identifiers = sorted(identifiers)
            self.num_identifiers = len(self.identifiers)

        self.capacity = capacity
        if embedding_size is None:
            self.embedding_size = 2 * self.capacity
        else:
            self.embedding_size = embedding_size

    def __str__(self):
        string = super().__str__()
        string += '{}-{}'.format(self.num_identifiers, self.embedding_size)
        return string

    def specification(self):
        spec = super().specification()
        spec.update(identifiers=self.identifiers, embedding_size=self.embedding_size)
        return spec

    def input_size(self):
        return self.embedding_size

    def output_size(self):
        return 0

    def extract(self, data):
        if self.identifiers is None:
            self.identifiers = sorted(data[self.name].unique())
            self.num_identifiers = len(self.identifiers)
        elif sorted(data[self.name].unique()) != self.identifiers:
            raise NotImplementedError

    def encode(self, data):
        if not isinstance(self.identifiers, int):
            data.loc[:, self.name] = data[self.name].map(arg=self.identifiers.index)
        data.loc[:, self.name] = data[self.name].astype(dtype='int64')
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

    def module_initialize(self):
        super().module_initialize()
        self.placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='input')
        # tf.placeholder_with_default(input=(-1,), shape=(None,), name='input')
        assert self.name not in Module.placeholders
        Module.placeholders[self.name] = self.placeholder
        initializer = util.get_initializer(initializer='normal-large')
        self.embeddings = tf.get_variable(
            name='embeddings', shape=(self.num_identifiers, self.embedding_size), dtype=tf.float32,
            initializer=initializer, regularizer=None, trainable=False, collections=None,
            caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
            custom_getter=None
        )
        self.current_identifier = tf.get_variable(
            name='current-identifier', shape=(), dtype=tf.int64, trainable=False
        )

    @tensorflow_name_scoped
    def input_tensor(self, feed=None):
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

    @tensorflow_name_scoped
    def next_identifier(self):
        assignment = self.current_identifier.assign_add(delta=1)
        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf.expand_dims(input=self.current_identifier, axis=0)

    @tensorflow_name_scoped
    def next_identifier_embedding(self):
        x = tf.random.normal(
            shape=(1, self.embedding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        return self.next_identifier(), x

    @tensorflow_name_scoped
    def random_value(self, n):
        identifier = tf.random_uniform(
            shape=(n,), minval=0, maxval=self.num_identifiers, dtype=tf.int32, seed=None
        )
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=identifier, partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        return identifier, x
