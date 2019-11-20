from typing import List

import numpy as np
import tensorflow as tf

from .value import Value
from .continuous import ContinuousValue
from .categorical import compute_embedding_size
from .. import util
from ..module import tensorflow_name_scoped


class NanValue(Value):

    def __init__(
        self, name: str, value: Value, capacity: int, weight_decay: float, weight: float,
        embedding_size: int = None, produce_nans: bool = False
    ):
        super().__init__(name=name)

        assert isinstance(value, ContinuousValue)
        # assert isinstance(value, (CategoricalValue, ContinuousValue))
        self.value = value

        self.capacity = capacity
        if embedding_size is None:
            embedding_size = compute_embedding_size(2, self.capacity)
        self.embedding_size = embedding_size
        self.weight_decay = weight_decay
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
            embedding_size=self.embedding_size, weight_decay=self.weight_decay
        )
        return spec

    def learned_input_columns(self):
        yield from self.value.learned_input_columns()

    def learned_output_columns(self):
        yield from self.value.learned_output_columns()

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

    def preprocess(self, df):
        if df[self.value.name].dtype.kind not in self.value.pd_types:
            df.loc[:, self.value.name] = self.value.pd_cast(df[self.value.name])
        nan = df[self.value.name].isna()
        df.loc[:, self.value.name] = df[self.value.name].fillna(method='bfill').fillna(method='ffill')
        # clean = df.dropna(subset=(self.value.name,))
        df = self.value.preprocess(df=df)
        # df.loc[:, self.value.name] = df[self.value.name].astype(encoded[self.value.name].dtype)
        # df = pd.merge(df.fillna(), encoded, how='outer')
        df.loc[nan, self.value.name] = np.nan
        return super().preprocess(df=df)

    def postprocess(self, df):
        df = super().postprocess(df=df)
        # clean = df.dropna(subset=(self.value.name,))
        # postprocessed = self.value.postprocess(data=clean)
        # df = pd.merge(data, postprocessed, how='outer')
        nan = df[self.value.name].isna()
        if nan.any():
            df.loc[:, self.value.name] = df[self.value.name].fillna(method='bfill').fillna(method='ffill')
        df = self.value.postprocess(df=df)
        if nan.any():
            df.loc[nan, self.value.name] = np.nan
        return df

    def module_initialize(self):
        super().module_initialize()

        shape = (2, self.embedding_size)
        initializer = util.get_initializer(initializer='normal')
        regularizer = util.get_regularizer(regularizer='l2', weight=self.weight_decay)
        self.embeddings = tf.compat.v1.get_variable(
            name='nan-embeddings', shape=shape, dtype=tf.float32, initializer=initializer,
            regularizer=regularizer, trainable=True, collections=None, caching_device=None,
            partitioner=None, validate_shape=True, use_resource=None, custom_getter=None
        )

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        return self.value.input_tensors()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 1

        # NaN embedding
        nan = tf.math.is_nan(x=xs[0])
        embedding = tf.nn.embedding_lookup(
            params=self.embeddings, ids=tf.cast(x=nan, dtype=tf.int64)
        )

        # Wrapped value input
        x = self.value.unify_inputs(xs=xs)

        # Set NaNs to zero to avoid propagating NaNs (which corresponds to mean because of quantile transformation)
        x = tf.where(condition=nan, x=tf.zeros_like(tensor=x), y=x)

        # Concatenate NaN embedding and wrapped value
        x = tf.concat(values=(embedding, x), axis=1)

        return x

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        # NaN classification part
        nan = tf.math.equal(x=tf.argmax(input=y[:, :2], axis=1), y=1)

        # Wrapped value output tensors
        ys = self.value.output_tensors(y=y[:, 2:])

        if self.produce_nans:
            # Replace wrapped value with NaNs
            for n, y in enumerate(ys):
                ys[n] = tf.where(condition=nan, x=(y * np.nan), y=y)

        return ys

    @tensorflow_name_scoped
    def loss(self, y, xs: List[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 1

        target = xs[0]
        target_nan = tf.math.is_nan(x=target)
        target_embedding = tf.one_hot(
            indices=tf.cast(x=target_nan, dtype=tf.int64), depth=2, on_value=1.0, off_value=0.0,
            axis=1, dtype=tf.float32
        )
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=target_embedding, logits=y[:, :2], axis=1
        )
        loss = self.weight * tf.reduce_mean(input_tensor=loss, axis=0)
        loss += self.value.loss(y=y[:, 2:], xs=xs, mask=tf.math.logical_not(x=target_nan))
        return loss
