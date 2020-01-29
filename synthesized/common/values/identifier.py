from typing import List

import pandas as pd
import tensorflow as tf

from .value import Value
from .. import util
from ..module import tensorflow_name_scoped


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

        self.embeddings = None
        self.placeholder = None
        self.current_identifier = None

    def __str__(self):
        string = super().__str__()
        string += '{}-{}'.format(self.num_identifiers, self.embedding_size)
        return string

    def specification(self):
        spec = super().specification()
        spec.update(identifiers=self.identifiers, embedding_size=self.embedding_size)
        return spec

    def learned_input_size(self) -> int:
        assert self.embedding_size is not None
        return self.embedding_size

    def learned_output_size(self):
        return 0

    def extract(self, df):
        super().extract(df=df)

        if self.identifiers is None:
            self.identifiers = sorted(df[self.name].unique())
            self.num_identifiers = len(self.identifiers)

        elif sorted(df[self.name].unique()) != self.identifiers:
            raise NotImplementedError

    def preprocess(self, df: pd.DataFrame):
        if not isinstance(self.identifiers, int):
            df.loc[:, self.name] = df[self.name].map(arg=self.identifiers.index)
        df.loc[:, self.name] = df[self.name].astype(dtype='int64')
        return super().preprocess(df)

    def module_initialize(self):
        super().module_initialize()
        # Input placeholder for value
        self.placeholder_initialize(dtype=tf.int64, shape=(None,))

        initializer = util.get_initializer(initializer='normal-large')
        shape = (self.num_identifiers, self.embedding_size)
        self.embeddings = tf.Variable(
            initial_value=initializer(shape=shape, dtype=tf.float32), name='embeddings', shape=shape,
            dtype=tf.float32, trainable=False, caching_device=None, validate_shape=True
        )
        self.current_identifier = tf.Variable(
            name='current-identifier', shape=(), dtype=tf.int64, trainable=False
        )

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        return [self.placeholder]

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 1
        assignment = self.current_identifier.assign(
            value=tf.maximum(x=self.current_identifier, y=tf.reduce_max(input_tensor=xs[0]))
        )
        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf.nn.embedding_lookup(params=self.embeddings, ids=xs[0])

    @tensorflow_name_scoped
    def next_identifier(self):
        assignment = self.current_identifier.assign_add(delta=1)
        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf.expand_dims(input=self.current_identifier, axis=0)

    @tensorflow_name_scoped
    def next_identifier_embedding(self):
        x = tf.random.normal(
            shape=(self.embedding_size,), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        return self.next_identifier(), x

    @tensorflow_name_scoped
    def random_value(self, n):
        identifier = tf.random.uniform(
            shape=(n,), minval=0, maxval=self.num_identifiers, dtype=tf.int32, seed=None
        )
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=identifier,
            max_norm=None
        )
        return identifier, x
