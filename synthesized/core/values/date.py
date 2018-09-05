import pandas as pd
import tensorflow as tf

from .value import Value
from .categorical import CategoricalValue
from .continuous import ContinuousValue


class DateValue(Value):

    def __init__(self, name, embedding_size, start_date=None):
        super().__init__(name=name)

        self.embedding_size = embedding_size
        self.start_date = start_date

        self.delta = self.add_module(
            module=ContinuousValue, name=self.name, positive=True
        )
        self.hour = self.add_module(
            module=CategoricalValue, name=(self.name + '-hour'), categories=24,
            embedding_size=embedding_size
        )
        self.dow = self.add_module(
            module=CategoricalValue, name=(self.name + '-dow'), categories=7,
            embedding_size=embedding_size
        )
        self.day = self.add_module(
            module=CategoricalValue, name=(self.name + '-day'), categories=31,
            embedding_size=embedding_size
        )
        self.month = self.add_module(
            module=CategoricalValue, name=(self.name + '-month'), categories=12,
            embedding_size=embedding_size
        )

    def specification(self):
        spec = super().specification()
        spec.update(embedding_size=self.embedding_size)
        return spec

    def input_size(self):
        return self.delta.input_size() + self.hour.input_size() + self.dow.input_size() + \
            self.day.input_size() + self.month.input_size()

    def output_size(self):
        return self.delta.output_size()

    def trainable_labels(self):
        yield from self.delta.trainable_labels()
        yield from self.hour.trainable_labels()
        yield from self.dow.trainable_labels()
        yield from self.day.trainable_labels()
        yield from self.month.trainable_labels()

    def placeholders(self):
        yield from self.delta.placeholders()
        yield from self.hour.placeholders()
        yield from self.dow.placeholders()
        yield from self.day.placeholders()
        yield from self.month.placeholders()

    def extract(self, data):
        if self.start_date is None:
            self.start_date = data[self.name].astype(dtype='datetime64').min()
        elif data[self.name].astype(dtype='datetime64').min() == self.start_date:
            raise NotImplementedError

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
            return tf.FixedLenFeature(shape=(), dtype=tf.float32, default_value=None)
        else:
            return tf.train.Feature(float_list=tf.train.FloatList(value=(x,)))

    def tf_input_tensor(self, feed=None):
        # python function????
        assert feed is None
        delta = self.delta.input_tensor(feed=feed)
        hour = self.hour.input_tensor(feed=feed)
        dow = self.dow.input_tensor(feed=feed)
        day = self.day.input_tensor(feed=feed)
        month = self.month.input_tensor(feed=feed)
        x = tf.concat(values=(delta, hour, dow, day, month), axis=1)
        return x

    def tf_output_tensors(self, x):
        return self.delta.output_tensor(x=x)

    def tf_loss(self, x, feed=None):
        # skip last and assume absolute value
        return self.delta.loss(x=x, feed=feed)
