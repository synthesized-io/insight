from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .categorical import compute_embedding_size
from .continuous import ContinuousValue
from .value import Value
from .. import util
from ..module import tensorflow_name_scoped


class NanValue(Value):

    def __init__(
        self, name: str, value: Value, capacity: int, weight: float,
        embedding_size: int = None, produce_nans: bool = False, produce_infs: bool = False
    ):
        super().__init__(name=name)

        assert isinstance(value, ContinuousValue)
        # assert isinstance(value, (CategoricalValue, ContinuousValue))
        self.value = value

        self.capacity = capacity
        if embedding_size is None:
            embedding_size = compute_embedding_size(4, similarity_based=False)
        self.embedding_size = embedding_size
        self.embedding_initialization = 'orthogonal-small'
        self.embeddings: Optional[tf.Variable] = None
        self.weight = weight

        self.produce_nans = produce_nans
        self.produce_infs = produce_infs

    def __str__(self):
        string = super().__str__()
        string += '-' + str(self.value)
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            value=self.value.specification(), produce_nans=self.produce_nans,
            embedding_size=self.embedding_size
        )
        return spec

    def learned_input_columns(self):
        return self.value.learned_input_columns()

    def learned_output_columns(self):
        return self.value.learned_output_columns()

    def learned_input_size(self):
        return self.embedding_size + self.value.learned_input_size()

    def learned_output_size(self):
        return 4 + self.value.learned_output_size()

    def extract(self, df):
        column = df[self.value.name]
        if column.dtype.kind not in self.value.pd_types:
            column = self.value.pd_cast(column)
        df_clean = df[~column.isin([np.NaN, pd.NaT, np.Inf, -np.Inf])]
        self.value.extract(df=df_clean)

        shape = (4, self.embedding_size)
        initializer = util.get_initializer(initializer='normal')
        self.embeddings = tf.Variable(
            initial_value=initializer(shape=shape, dtype=tf.float32), name='nan-embeddings', shape=shape,
            dtype=tf.float32, trainable=True, caching_device=None, validate_shape=True
        )
        self.add_regularization_weight(self.embeddings)

    @tensorflow_name_scoped
    def build(self) -> None:
        if not self.built:
            shape = (4, self.embedding_size)
            initializer = util.get_initializer(initializer='normal')
            self.embeddings = tf.Variable(
                initial_value=initializer(shape=shape, dtype=tf.float32), name='nan-embeddings', shape=shape,
                dtype=tf.float32, trainable=True, caching_device=None, validate_shape=True
            )
            self.add_regularization_weight(self.embeddings)

        self.built = True

    def preprocess(self, df):
        df.loc[:, self.value.name] = pd.to_numeric(df.loc[:, self.value.name], errors='coerce')

        nan_inf = df.loc[:, self.value.name].isin([np.NaN, pd.NaT, np.Inf, -np.Inf])
        df.loc[~nan_inf, :] = self.value.preprocess(df=df.loc[~nan_inf, :])
        df.loc[:, self.value.name] = df.loc[:, self.value.name].astype(np.float32)

        return super().preprocess(df=df)

    def postprocess(self, df):
        df = super().postprocess(df=df)

        nan_inf = df.loc[:, self.value.name].isin([np.NaN, pd.NaT, np.Inf, -np.Inf])
        df.loc[~nan_inf, :] = self.value.postprocess(df=df.loc[~nan_inf, :])

        return df

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        # NaN embedding
        nan = tf.math.is_nan(x=xs[0])
        inf = tf.math.is_inf(x=xs[0])
        pos = tf.greater(xs[0], tf.constant(0.0, dtype=tf.float32))

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
    def output_tensors(self, y: tf.Tensor, sample: bool = True, **kwargs) -> List[tf.Tensor]:
        # NaN classification part
        if sample:
            nan = tf.squeeze(tf.random.categorical(logits=y[:, :4], num_samples=1), axis=1)
        else:
            nan = tf.argmax(input=y[:, :4], axis=1)

        # Wrapped value output tensors
        ys = self.value.output_tensors(y=y[:, 4:], **kwargs)

        if self.produce_nans:
            # Replace wrapped value with NaNs
            for n, y in enumerate(ys):
                ys[n] = tf.where(condition=tf.math.equal(x=nan, y=1), x=np.nan, y=y)

        if self.produce_infs:
            # Replace wrapped value with Infs
            for n, y in enumerate(ys):
                ys[n] = tf.where(condition=tf.math.equal(x=nan, y=2), x=np.inf, y=y)
                ys[n] = tf.where(condition=tf.math.equal(x=nan, y=3), x=-np.inf, y=y)

        return ys

    @tensorflow_name_scoped
    def loss(self, y, xs: List[tf.Tensor]) -> tf.Tensor:
        target = xs[0]
        target_nan = tf.logical_or(tf.math.is_nan(x=target), tf.math.is_inf(x=target))
        target_embedding = tf.one_hot(
            indices=tf.cast(x=target_nan, dtype=tf.int64), depth=4, on_value=1.0, off_value=0.0,
            axis=1, dtype=tf.float32
        )
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_embedding, logits=y[:, :4], axis=1
        )
        loss = self.weight * tf.reduce_mean(input_tensor=loss, axis=0)
        loss += self.value.loss(y=y[:, 4:], xs=xs, mask=tf.math.logical_not(x=target_nan))
        tf.summary.scalar(name=self.name, data=loss)
        return loss
