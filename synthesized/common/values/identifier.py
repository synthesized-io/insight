from typing import Dict, Optional

import tensorflow as tf

from .value import Value
from ..util import get_initializer
from ..module import tensorflow_name_scoped
from ...config import IdentifierConfig


class IdentifierValue(Value):
    def __init__(
        self, name, num_identifiers=None, embedding_size=None, config: IdentifierConfig = IdentifierConfig()
    ):
        super().__init__(name=name)
        self.num_identifiers = num_identifiers
        self.identifier2idx: Optional[Dict] = None

        self.capacity = config.capacity
        if embedding_size is None:
            self.embedding_size = self.capacity
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
        assert self.num_identifiers is not None
        return self.num_identifiers

    @tensorflow_name_scoped
    def build(self) -> None:
        if not self.built:
            initializer = get_initializer(initializer='glorot-normal')
            shape = (self.num_identifiers, self.embedding_size)
            self.embeddings = tf.Variable(
                initial_value=initializer(shape=shape, dtype=tf.float32), name='embeddings', shape=shape,
                dtype=tf.float32, trainable=False, caching_device=None, validate_shape=True
            )
            self.current_identifier = tf.Variable(
                initial_value=0, name='current-identifier', shape=(), dtype=tf.int64, trainable=False
            )

        self.built = True

    @tensorflow_name_scoped
    def unify_inputs(self, xs: tf.Tensor) -> tf.Tensor:
        self.build()
        x = tf.cast(tf.squeeze(xs, axis=-1), dtype=tf.int64)
        return tf.nn.embedding_lookup(params=self.embeddings, ids=x)

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
    def random_value(self):
        identifier = tf.random.uniform(
            shape=(1,), minval=0, maxval=self.num_identifiers, dtype=tf.int32, seed=None
        )
        identifier_embedding = tf.nn.embedding_lookup(params=self.embeddings, ids=identifier)
        return identifier, identifier_embedding

    @tensorflow_name_scoped
    def random_value_from_normal(self):
        identifier_embedding = tf.random.normal(
            shape=(self.embedding_size, 1), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        identifier = tf.argmax(tf.linalg.matmul(self.embeddings, identifier_embedding))

        return identifier, identifier_embedding

    @tensorflow_name_scoped
    def get_embedding(self, identifier: tf.Tensor):
        return tf.nn.embedding_lookup(params=self.embeddings, ids=identifier)

    # def loss(self, y: tf.Tensor, xs: tf.Tensor) -> tf.Tensor:
    #     target = tf.one_hot(
    #         indices=xs[:, 0], depth=self.num_identifiers, on_value=1.0, off_value=0.0, axis=1,
    #         dtype=tf.float32
    #     )
    #
    #     loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y, axis=1)
    #     loss = self.weight * tf.reduce_mean(input_tensor=loss, axis=0)
    #     tf.summary.scalar(name=self.name, data=loss)
    #     return loss
