from math import log
import numpy as np
import pandas as pd
import tensorflow as tf

from .value import Value
from .. import util
from ..module import Module


class NanValue(Value):

    def __init__(self, name, value, capacity=None, embedding_size=None, weight_decay=0.0):
        super().__init__(name=name)

        self.value = value
        self.weight_decay = weight_decay

        self.capacity = capacity
        if embedding_size is None:
            self.embedding_size = int(log(2) * self.capacity / 2.0)
        else:
            self.embedding_size = embedding_size
        self.weight_decay = weight_decay

    def __str__(self):
        string = super().__str__()
        string += '{}-{}'.format(self.embedding_size, self.value)
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            value=self.value.specification(), embedding_size=self.embedding_size,
            weight_decay=self.weight_decay
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
        clean = data.dropna(subset=(self.value.name,))
        self.value.extract(data=clean)

    def encode(self, data):
        clean = data.dropna(subset=(self.value.name,))
        encoded = self.value.encode(data=clean)
        data = pd.merge(data, encoded, how='outer')
        return data

    def postprocess(self, data):
        clean = data.dropna(subset=(self.value.name,))
        postprocessed = self.value.postprocess(data=clean)
        data = pd.merge(data, postprocessed, how='outer')
        return data

    def features(self, x=None):
        return self.value.features(x=x)

    def tf_initialize(self):
        super().tf_initialize()

        shape = (2, self.embedding_size)
        initializer = util.get_initializer(initializer='normal')
        regularizer = util.get_regularizer(regularizer='l2', weight=self.weight_decay)
        self.embeddings = tf.get_variable(
            name='embeddings', shape=shape, dtype=tf.float32, initializer=initializer,
            regularizer=regularizer, trainable=True, collections=None, caching_device=None,
            partitioner=None, validate_shape=True, use_resource=None, custom_getter=None
        )

    def tf_input_tensor(self, feed=None):
        x = self.value.placeholder if feed is None else feed[self.value.name]
        nan = tf.cast(x=tf.is_nan(x=x), dtype=tf.int64)
        embedding = tf.nn.embedding_lookup(
            params=self.embeddings, ids=nan, partition_strategy='mod', validate_indices=True,
            max_norm=None
        )
        x = self.value.input_tensor(feed=feed)
        x = tf.where(condition=nan, x=0.0, y=x)
        x = tf.expand_dims(input=x, axis=1)
        x = tf.concat(values=(embedding, x), axis=1)
        return x

    def tf_output_tensors(self, x):
        nan = tf.argmax(input=x[:, :2], axis=1)
        x = self.value.output_tensors(x=x[:, 2:])[self.value.name]
        x = tf.where(condition=nan, x=np.nan, y=x)
        return {self.name: x}

    def tf_loss(self, x, feed=None):
        target = self.value.placeholder if feed is None else feed[self.value.name]
        target_nan = tf.cast(x=tf.is_nan(x=target), dtype=tf.int64)
        target_nan = tf.one_hot(
            indices=target_nan, depth=2, on_value=1.0, off_value=0.0, axis=1, dtype=tf.float32
        )
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target_nan, logits=x[:, :2], weights=None, label_smoothing=None,
            scope=None, loss_collection=tf.GraphKeys.LOSSES
        )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        loss += tf.where(condition=target_nan, x=0.0, y=self.value.loss(x=x[:, 2:], feed=feed))
        return loss