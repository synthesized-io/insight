import pandas as pd
import tensorflow as tf

from .value import Value
from .categorical import CategoricalValue
from .continuous import ContinuousValue


class DateValue(Value):

    def __init__(self, name, capacity=None, embedding_size=None, start_date=None):
        super().__init__(name=name)

        self.start_date = start_date

        self.delta = self.add_module(
            module=ContinuousValue, name=self.name, positive=True
        )
        self.hour = self.add_module(
            module=CategoricalValue, name=(self.name + '-hour'), categories=24, capacity=capacity,
            embedding_size=embedding_size
        )
        self.dow = self.add_module(
            module=CategoricalValue, name=(self.name + '-dow'), categories=7, capacity=capacity,
            embedding_size=embedding_size
        )
        self.day = self.add_module(
            module=CategoricalValue, name=(self.name + '-day'), categories=31, capacity=capacity,
            embedding_size=embedding_size
        )
        self.month = self.add_module(
            module=CategoricalValue, name=(self.name + '-month'), categories=12, capacity=capacity,
            embedding_size=embedding_size
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            delta=self.delta.specification(), hour=self.hour.specification(),
            dow=self.dow.specification(), day=self.day.specification(),
            month=self.month.specification()
        )
        return spec

    def input_size(self):
        return self.delta.input_size() + self.hour.input_size() + self.dow.input_size() + \
            self.day.input_size() + self.month.input_size()

    def output_size(self):
        return self.delta.output_size()

    def input_labels(self):
        yield from self.delta.input_labels()
        yield from self.hour.input_labels()
        yield from self.dow.input_labels()
        yield from self.day.input_labels()
        yield from self.month.input_labels()

    def output_labels(self):
        yield from self.delta.output_labels()

    def placeholders(self):
        yield from self.delta.placeholders()
        yield from self.hour.placeholders()
        yield from self.dow.placeholders()
        yield from self.day.placeholders()
        yield from self.month.placeholders()

    def extract(self, data):
        # happens in identify_value
        # data[self.name] = pd.to_datetime(data[self.name])
        if self.start_date is None:
            # self.start_date = data[self.name].astype(dtype='datetime64').min()
            self.start_date = data[self.name].min()
        # elif data[self.name].astype(dtype='datetime64').min() == self.start_date:
        elif data[self.name].min() < self.start_date:
            raise NotImplementedError

    def encode(self, data):
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

    def features(self, x=None):
        features = super().features(x=x)
        features.update(self.delta.features(x=x))
        features.update(self.hour.features(x=x))
        features.update(self.dow.features(x=x))
        features.update(self.day.features(x=x))
        features.update(self.month.features(x=x))
        return features

    def tf_input_tensor(self, feed=None):
        delta = self.delta.input_tensor(feed=feed)
        hour = self.hour.input_tensor(feed=feed)
        dow = self.dow.input_tensor(feed=feed)
        day = self.day.input_tensor(feed=feed)
        month = self.month.input_tensor(feed=feed)
        x = tf.concat(values=(delta, hour, dow, day, month), axis=1)
        return x

    def tf_output_tensors(self, x):
        return self.delta.output_tensors(x=x)

    def tf_loss(self, x, feed=None):
        # skip last and assume absolute value
        return self.delta.loss(x=x, feed=feed)
