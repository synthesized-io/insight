import pandas as pd
import tensorflow as tf

from .value import Value
from .. import util
from ..module import Module


class DateValue(Value):

    def __init__(self, name, embedding_size):
        super().__init__(name=name)
        self.embedding_size = embedding_size
        self.start_date = pd.to_datetime('1993-01-01 00:00:00')

    def specification(self):
        spec = super().specification()
        spec.update(embedding_size=self.embedding_size)
        return spec

    def input_size(self):
        return 1 + 4 * self.embedding_size

    def output_size(self):
        return 1

    def preprocess(self, data):
        data[self.name] = data[self.name].astype(dtype='datetime64')
        data = data.sort_values(by=self.name)
        data[self.name + '-hour'] = data[self.name].dt.hour
        data[self.name + '-dow'] = data[self.name].dt.weekday
        data[self.name + '-day'] = data[self.name].dt.day - 1
        data[self.name + '-month'] = data[self.name].dt.month - 1
        previous_date = data[self.name].copy()
        previous_date[0] = self.start_date
        previous_date[1:] = previous_date[:-1]
        data[self.name] = (data[self.name] - previous_date).dt.total_seconds() / (24 * 60 * 60)
        return data

    def postprocess(self, data):
        data[self.name] = pd.to_timedelta(arg=data[self.name], unit='D').cumsum(axis=0)
        data[self.name] += self.start_date
        return data

    def feature(self, x=None):
        if x is None:
            return tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)
        else:
            return tf.train.Feature(float_list=tf.train.Int64List(value=(x,)))

    def tf_initialize(self):
        super().tf_initialize()
        self.placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name='input')
        Module.placeholders[self.name] = self.placeholder
        self.hour_placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='hour')
        Module.placeholders[self.name + '-hour'] = self.hour_placeholder
        self.dow_placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='dow')
        Module.placeholders[self.name + '-dow'] = self.dow_placeholder
        self.day_placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='day')
        Module.placeholders[self.name + '-day'] = self.day_placeholder
        self.month_placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='month')
        Module.placeholders[self.name + '-month'] = self.month_placeholder
        self.hour_embeddings = tf.get_variable(
            name='hours', shape=(24, self.embedding_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.dow_embeddings = tf.get_variable(
            name='days-of-week', shape=(7, self.embedding_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.day_embeddings = tf.get_variable(
            name='days', shape=(31, self.embedding_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.month_embeddings = tf.get_variable(
            name='months', shape=(12, self.embedding_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )

    def tf_input_tensor(self, feed=None):
        # python function????
        assert feed is None
        timedelta = tf.expand_dims(input=self.placeholder, axis=1)
        hour = tf.nn.embedding_lookup(
            params=self.hour_embeddings, ids=self.hour_placeholder, partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        dow = tf.nn.embedding_lookup(
            params=self.dow_embeddings, ids=self.dow_placeholder, partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        day = tf.nn.embedding_lookup(
            params=self.day_embeddings, ids=self.day_placeholder, partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        month = tf.nn.embedding_lookup(
            params=self.month_embeddings, ids=self.month_placeholder, partition_strategy='mod',
            validate_indices=True, max_norm=None
        )
        x = tf.concat(values=(timedelta, hour, dow, day, month), axis=1)
        return x

    def tf_output_tensor(self, x):
        x = tf.nn.softplus(features=x)
        x = tf.squeeze(input=x, axis=1)
        return x

    def tf_loss(self, x, feed=None):
        # skip last and assume absolute value
        assert feed is None
        x = tf.nn.softplus(features=x)
        target = tf.expand_dims(input=self.placeholder, axis=1)
        loss = tf.losses.mean_squared_error(
            labels=target, predictions=x, weights=1.0, scope=None,
            loss_collection=tf.GraphKeys.LOSSES
        )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        return loss
