from math import log
import numpy as np
import pandas as pd
import tensorflow as tf

from .value import Value
# from .categorical import CategoricalValue
from .continuous import ContinuousValue
from ..module import tensorflow_name_scoped
from .. import util


class NanValue(Value):

    def __init__(
        self, name, value, produce_nans=False, capacity=None, embedding_size=None, weight_decay=0.0
    ):
        super().__init__(name=name)

        assert isinstance(value, ContinuousValue)
        # assert isinstance(value, (CategoricalValue, ContinuousValue))
        self.value = value
        self.produce_nans = produce_nans

        self.capacity = capacity
        if embedding_size is None:
            self.embedding_size = int(log(2) * self.capacity / 2.0)
        else:
            self.embedding_size = embedding_size
        self.weight_decay = weight_decay

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

    def input_size(self):
        return self.embedding_size + self.value.input_size()

    def output_size(self):
        return 2 + self.value.output_size()

    def input_labels(self):
        yield from self.value.input_labels()

    def output_labels(self):
        yield from self.value.output_labels()

    def placeholders(self):
        yield from self.value.placeholders()

    def extract(self, data):
        column = data[self.value.name]
        if column.dtype.kind not in self.value.pd_types:
            column = self.value.pd_cast(column)
        clean = data[column.notna()]
        self.value.extract(data=clean)

    def preprocess(self, data):
        if data[self.value.name].dtype.kind not in self.value.pd_types:
            data.loc[:, self.value.name] = self.value.pd_cast(data[self.value.name])
        nan = data[self.value.name].isna()
        data.loc[:, self.value.name] = data[self.value.name].fillna(method='bfill').fillna(method='ffill')
        # clean = data.dropna(subset=(self.value.name,))
        data = self.value.preprocess(data=data)
        # data.loc[:, self.value.name] = data[self.value.name].astype(encoded[self.value.name].dtype)
        # data = pd.merge(data.fillna(), encoded, how='outer')
        data.loc[nan, self.value.name] = np.nan
        return data

    def postprocess(self, data):
        # clean = data.dropna(subset=(self.value.name,))
        # postprocessed = self.value.postprocess(data=clean)
        # data = pd.merge(data, postprocessed, how='outer')
        nan = data[self.value.name].isna()
        if nan.any():
            data.loc[:, self.value.name] = data[self.value.name].fillna(method='bfill').fillna(method='ffill')
        data = self.value.postprocess(data=data)
        if nan.any():
            data.loc[nan, self.value.name] = np.nan
        return data

    def features(self, x=None):
        return self.value.features(x=x)

    def module_initialize(self):
        super().module_initialize()

        shape = (2, self.embedding_size)
        initializer = util.get_initializer(initializer='normal')
        regularizer = util.get_regularizer(regularizer='l2', weight=self.weight_decay)
        self.embeddings = tf.get_variable(
            name='nan-embeddings', shape=shape, dtype=tf.float32, initializer=initializer,
            regularizer=regularizer, trainable=True, collections=None, caching_device=None,
            partitioner=None, validate_shape=True, use_resource=None, custom_getter=None
        )

    @tensorflow_name_scoped
    def input_tensor(self, feed=None):
        x = self.value.placeholder if feed is None else feed[self.value.name]
        nan = tf.is_nan(x=x)
        embedding = tf.nn.embedding_lookup(
            params=self.embeddings, ids=tf.cast(x=nan, dtype=tf.int64), partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        x = self.value.input_tensor(feed=feed)
        x = tf.where(condition=nan, x=tf.zeros_like(tensor=x), y=x)
        # x = tf.expand_dims(input=x, axis=1)
        x = tf.concat(values=(embedding, x), axis=1)
        return x

    @tensorflow_name_scoped
    def output_tensors(self, x):
        nan = (tf.math.equal(x=tf.argmax(input=x[:, :2], axis=1), y=1))
        x = self.value.output_tensors(x=x[:, 2:])[self.value.name]
        if self.produce_nans:
            x = tf.where(condition=nan, x=(x * np.nan), y=x)
        return {self.value.name: x}

    @tensorflow_name_scoped
    def loss(self, x, feed=None):
        target = self.value.placeholder if feed is None else feed[self.value.name]
        target_nan = tf.is_nan(x=target)
        target_embedding = tf.one_hot(
            indices=tf.cast(x=target_nan, dtype=tf.int64), depth=2, on_value=1.0, off_value=0.0,
            axis=1, dtype=tf.float32
        )
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target_embedding, logits=x[:, :2], weights=1.0, label_smoothing=0,
            scope=None, loss_collection=tf.GraphKeys.LOSSES
        )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        loss += self.value.loss(x=x[:, 2:], feed=feed, mask=tf.math.logical_not(x=target_nan))
        return loss
