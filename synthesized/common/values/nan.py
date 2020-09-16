from typing import Sequence

import numpy as np
import tensorflow as tf

from .categorical import compute_embedding_size
from .continuous import ContinuousValue
from .value import Value
from ..util import get_initializer
from ..module import tensorflow_name_scoped
from ...config import NanConfig


class NanValue(Value):

    def __init__(
        self, name: str, value: Value, config: NanConfig = NanConfig(),
        embedding_size: int = None
    ):
        super().__init__(name=name)

        assert isinstance(value, ContinuousValue)
        # assert isinstance(value, (CategoricalValue, ContinuousValue))
        self.value = value
        self.num_categories = 4  # Num, NaN, Inf, -Inf

        if embedding_size is None:
            embedding_size = compute_embedding_size(self.num_categories, similarity_based=False)
        self.embedding_size = embedding_size
        self.embedding_initialization = 'orthogonal-small'
        self.weight = config.nan_weight

        shape = (self.num_categories, self.embedding_size)
        initializer = get_initializer(initializer='normal')
        self.embeddings = tf.Variable(
            initial_value=initializer(shape=shape, dtype=tf.float32), name='nan-embeddings', shape=shape,
            dtype=tf.float32, trainable=True, caching_device=None, validate_shape=True
        )
        self.add_regularization_weight(self.embeddings)

    def __str__(self):
        string = super().__str__()
        string += '-' + str(self.value)
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            value=self.value.specification(),
            embedding_size=self.embedding_size
        )
        return spec

    def learned_input_size(self):
        return self.embedding_size + self.value.learned_input_size()

    def learned_output_size(self):
        return self.num_categories + self.value.learned_output_size()

    @tensorflow_name_scoped
    def build(self) -> None:
        if not self.built:
            shape = (self.num_categories, self.embedding_size)
            initializer = get_initializer(initializer='normal')
            self.embeddings = tf.Variable(
                initial_value=initializer(shape=shape, dtype=tf.float32), name='nan-embeddings', shape=shape,
                dtype=tf.float32, trainable=True, caching_device=None, validate_shape=True
            )
            self.add_regularization_weight(self.embeddings)

        self.built = True

    @tensorflow_name_scoped
    def unify_inputs(self, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        # NaN embedding
        x = xs[0]
        nan = tf.math.is_nan(x=x)
        inf = tf.math.is_inf(x=x)
        pos = tf.greater(x, tf.constant(0.0, dtype=tf.float32))

        nan_int = tf.cast(nan, dtype=tf.int64)
        pos_inf = tf.constant(2, dtype=tf.int64) * tf.cast(tf.logical_and(inf, pos), dtype=tf.int64)
        neg_inf = tf.constant(3, dtype=tf.int64) * tf.cast(tf.logical_and(inf, tf.logical_not(pos)), dtype=tf.int64)

        embedding = tf.nn.embedding_lookup(
            params=self.embeddings,
            ids=nan_int + pos_inf + neg_inf
        )

        # Wrapped value input
        x = self.value.unify_inputs(xs=xs)

        # Set NaNs to random noise to avoid propagating NaNs
        x = tf.where(
            condition=tf.expand_dims(input=tf.logical_or(nan, inf), axis=1),
            x=tf.random.normal(shape=x.shape),
            y=x
        )

        # Concatenate NaN embedding and wrapped value
        x = tf.concat(values=(embedding, x), axis=1)
        return x

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, sample: bool = True, produce_nans: bool = False, produce_infs: bool = False,
                       **kwargs) -> Sequence[tf.Tensor]:
        # NaN classification part
        if sample:
            nan = tf.squeeze(tf.random.categorical(logits=y[:, :self.num_categories], num_samples=1), axis=-1)
        else:
            nan = tf.argmax(input=y[:, :self.num_categories], axis=1)

        # Wrapped value output tensors
        ys = list(self.value.output_tensors(y=y[:, self.num_categories:], **kwargs))

        for n, y in enumerate(ys):
            if produce_nans:
                # Replace wrapped value with NaNs
                ys[n] = tf.where(condition=tf.math.equal(x=nan, y=1), x=np.nan, y=ys[n])

            if produce_infs:
                # Replace wrapped value with Infs
                ys[n] = tf.where(condition=tf.math.equal(x=nan, y=2), x=np.inf, y=ys[n])
                ys[n] = tf.where(condition=tf.math.equal(x=nan, y=3), x=-np.inf, y=ys[n])

        return tuple(ys)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        target = xs[0]
        nan = tf.math.is_nan(x=target)
        inf = tf.math.is_inf(x=target)
        nan_or_inf = tf.logical_or(x=nan, y=inf)
        pos = tf.greater(target, tf.constant(0.0, dtype=tf.float32))

        nan_int = tf.cast(nan, dtype=tf.int64)
        pos_inf = tf.constant(2, dtype=tf.int64) * tf.cast(tf.logical_and(inf, pos), dtype=tf.int64)
        neg_inf = tf.constant(3, dtype=tf.int64) * tf.cast(tf.logical_and(inf, tf.logical_not(pos)), dtype=tf.int64)

        target_nan = nan_int + pos_inf + neg_inf
        target_embedding = tf.one_hot(
            indices=tf.cast(x=target_nan, dtype=tf.int64), depth=self.num_categories, on_value=1.0, off_value=0.0,
            axis=1, dtype=tf.float32
        )
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_embedding, logits=y[:, :self.num_categories], axis=1
        )
        loss = self.weight * tf.reduce_mean(input_tensor=loss, axis=0)
        loss += self.value.loss(y=y[:, self.num_categories:], xs=xs, mask=tf.math.logical_not(x=nan_or_inf))
        tf.summary.scalar(name=self.name, data=loss)
        return loss
