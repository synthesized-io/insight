from typing import List, Optional, Dict

import pandas as pd
import tensorflow as tf

from .value import Value
from .. import util
from ..module import tensorflow_name_scoped


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

        self.identifier2idx: Optional[Dict] = None

        self.capacity = capacity
        self.embedding_size = embedding_size
        if embedding_size is None:
            self.embedding_size = 2 * self.capacity
        else:
            self.embedding_size = embedding_size

        self.embeddings = None
        self.placeholder = None
        # self.current_identifier = None

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

    def learned_output_size(self) -> int:
        return 1

    def extract(self, df):
        super().extract(df=df)

        if self.identifiers is None:
            self.identifiers = sorted(df.loc[:, self.name].unique())
            self.num_identifiers = len(self.identifiers)
        elif sorted(df.loc[:, self.name].unique()) != self.identifiers:
            raise NotImplementedError

        self.identifier2idx = {k: i for i, k in enumerate(self.identifiers)}
        self.idx2identifier = {i: k for i, k in enumerate(self.identifiers)}

        self.build()

    @tensorflow_name_scoped
    def build(self) -> None:
        if not self.built:
            initializer = util.get_initializer(initializer='normal-large')
            shape = (self.num_identifiers, self.embedding_size)
            self.embeddings = tf.Variable(
                initial_value=initializer(shape=shape, dtype=tf.float32), name='embeddings', shape=shape,
                dtype=tf.float32, trainable=False, caching_device=None, validate_shape=True
            )
            self.current_identifier = tf.Variable(
                initial_value=0, name='current-identifier', shape=(), dtype=tf.int64, trainable=False
            )

        self.built = True

    def preprocess(self, df: pd.DataFrame):
        df.loc[:, self.name] = df.loc[:, self.name].map(self.identifier2idx)
        if df.loc[:, self.name].dtype != 'int64':
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='int64')
        return super().preprocess(df)

    def postprocess(self, df: pd.DataFrame):
        df = super().postprocess(df=df)
        df.loc[:, self.name] = df.loc[:, self.name].map(self.idx2identifier)
        return df

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 1
        self.build()
        return tf.nn.embedding_lookup(params=self.embeddings, ids=tf.cast(xs[0], dtype=tf.int64))

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
