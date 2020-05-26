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
        embedding_size: int = None, produce_nans: bool = False
    ):
        super().__init__(name=name)

        assert isinstance(value, ContinuousValue)
        # assert isinstance(value, (CategoricalValue, ContinuousValue))
        self.value = value

        self.capacity = capacity
        if embedding_size is None:
            embedding_size = compute_embedding_size(2, similarity_based=False)
        self.embedding_size = embedding_size
        self.embedding_initialization = 'orthogonal-small'
        self.embeddings: Optional[tf.Variable] = None
        self.weight = weight

        self.produce_nans = produce_nans

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
        return 2 + self.value.learned_output_size()

    def extract(self, df):
        column = df[self.value.name]
        if column.dtype.kind not in self.value.pd_types:
            column = self.value.pd_cast(column)
        df_clean = df[column.notna()]
        self.value.extract(df=df_clean)

        shape = (2, self.embedding_size)
        initializer = util.get_initializer(initializer='normal')
        self.embeddings = tf.Variable(
            initial_value=initializer(shape=shape, dtype=tf.float32), name='nan-embeddings', shape=shape,
            dtype=tf.float32, trainable=True, caching_device=None, validate_shape=True
        )
        self.add_regularization_weight(self.embeddings)

    @tensorflow_name_scoped
    def build(self) -> None:
        if not self.built:
            shape = (2, self.embedding_size)
            initializer = util.get_initializer(initializer='normal')
            self.embeddings = tf.Variable(
                initial_value=initializer(shape=shape, dtype=tf.float32), name='nan-embeddings', shape=shape,
                dtype=tf.float32, trainable=True, caching_device=None, validate_shape=True
            )
            self.add_regularization_weight(self.embeddings)

        self.built = True

    def preprocess(self, df):
        df.loc[:, self.value.name] = pd.to_numeric(df.loc[:, self.value.name], errors='coerce')

        nan = df.loc[:, self.value.name].isna()
        df.loc[~nan, :] = self.value.preprocess(df=df.loc[~nan, :])
        df.loc[:, self.value.name] = df.loc[:, self.value.name].astype(np.float32)

        return super().preprocess(df=df)

    def postprocess(self, df):
        df = super().postprocess(df=df)

        nan = df.loc[:, self.value.name].isna()
        df.loc[~nan, :] = self.value.postprocess(df=df.loc[~nan, :])

        return df

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        # NaN embedding
        nan = tf.math.is_nan(x=xs[0])
        embedding = tf.nn.embedding_lookup(
            params=self.embeddings, ids=tf.cast(x=nan, dtype=tf.int64)
        )

        # Wrapped value input
        x = self.value.unify_inputs(xs=xs)

        # Set NaNs to zero to avoid propagating NaNs (which corresponds to mean because of quantile transformation)
        x = tf.where(condition=tf.expand_dims(input=nan, axis=1), x=tf.zeros_like(input=x), y=x)

        # Concatenate NaN embedding and wrapped value
        x = tf.concat(values=(embedding, x), axis=1)
        return x

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        sample: bool = kwargs['sample'] if 'sample' in kwargs.keys() else False
        # NaN classification part
        if sample:
            nan = tf.math.equal(x=tf.argmax(input=y[:, :2], axis=1), y=1)
        else:
            nan = tf.math.equal(x=tf.squeeze(tf.random.categorical(logits=y[:, :2], num_samples=1), axis=1), y=1)

        # Wrapped value output tensors
        ys = self.value.output_tensors(y=y[:, 2:], **kwargs)

        if self.produce_nans:
            # Replace wrapped value with NaNs
            for n, y in enumerate(ys):
                ys[n] = tf.where(condition=nan, x=(y * np.nan), y=y)

        return ys

    @tensorflow_name_scoped
    def loss(self, y, xs: List[tf.Tensor]) -> tf.Tensor:
        target = xs[0]
        target_nan = tf.math.is_nan(x=target)
        target_embedding = tf.one_hot(
            indices=tf.cast(x=target_nan, dtype=tf.int64), depth=2, on_value=1.0, off_value=0.0,
            axis=1, dtype=tf.float32
        )
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_embedding, logits=y[:, :2], axis=1
        )
        loss = self.weight * tf.reduce_mean(input_tensor=loss, axis=0)
        loss += self.value.loss(y=y[:, 2:], xs=xs, mask=tf.math.logical_not(x=target_nan))
        tf.summary.scalar(name=self.name, data=loss)
        return loss
