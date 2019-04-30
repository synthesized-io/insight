import pandas as pd
import tensorflow as tf

from .categorical import CategoricalValue
from .continuous import ContinuousValue
from ..module import tensorflow_name_scoped


class DateValue(ContinuousValue):

    def __init__(self, name, capacity=None, embedding_size=None, start_date=None, min_date=None):
        super().__init__(name=name)

        assert start_date is None or min_date is None
        self.start_date = start_date
        self.min_date = min_date

        self.pd_types = ('M',)
        self.pd_cast = (lambda x: pd.to_datetime(x))

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

    def __str__(self):
        string = super().__str__()
        if self.start_date is not None:
            string += '-timedelta'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            hour=self.hour.specification(), dow=self.dow.specification(),
            day=self.day.specification(), month=self.month.specification()
        )
        return spec

    def input_size(self):
        return super().input_size() + self.hour.input_size() + self.dow.input_size() + \
            self.day.input_size() + self.month.input_size()

    def output_size(self):
        return super().output_size()

    def input_labels(self):
        yield from super().input_labels()
        yield from self.hour.input_labels()
        yield from self.dow.input_labels()
        yield from self.day.input_labels()
        yield from self.month.input_labels()

    def placeholders(self):
        yield from super().placeholders()
        yield from self.hour.placeholders()
        yield from self.dow.placeholders()
        yield from self.day.placeholders()
        yield from self.month.placeholders()

    def extract(self, data):
        column = data[self.name]

        if column.dtype.kind != 'M':
            column = pd.to_datetime(column)

        if column.is_monotonic:
            if self.start_date is None:
                self.start_date = column[0] - (column.values[1:] - column.values[:-1]).mean()
            elif column[0] < self.start_date:
                raise NotImplementedError
            previous_date = column.values.copy()
            previous_date[1:] = previous_date[:-1]
            previous_date[0] = self.start_date
            column = (column - previous_date).dt.total_seconds() / (24 * 60 * 60)

        else:
            if self.min_date is None:
                self.min_date = column.min()
            elif column.min() != self.min_date:
                raise NotImplementedError
            column = (column - self.min_date).dt.total_seconds() / (24 * 60 * 60)

        super().extract(data=pd.DataFrame.from_dict({self.name: column}))

    def preprocess(self, data):
        if data[self.name].dtype.kind != 'M':
            data.loc[:, self.name] = pd.to_datetime(data[self.name])
        data.loc[:, self.name + '-hour'] = data[self.name].dt.hour
        data.loc[:, self.name + '-dow'] = data[self.name].dt.weekday
        data.loc[:, self.name + '-day'] = data[self.name].dt.day - 1
        data.loc[:, self.name + '-month'] = data[self.name].dt.month - 1
        if self.min_date is None:
            previous_date = data[self.name].copy()
            previous_date[0] = self.start_date
            previous_date[1:] = previous_date[:-1]
            data.loc[:, self.name] = (data[self.name] - previous_date).dt.total_seconds() / (24 * 60 * 60)
        else:
            data.loc[:, self.name] = (data[self.name] - self.min_date).dt.total_seconds() / (24 * 60 * 60)
        return super().preprocess(data=data)

    def postprocess(self, data):
        data = super().postprocess(data=data)
        data.loc[:, self.name] = pd.to_timedelta(arg=data[self.name], unit='D')
        if self.start_date is not None:
            data.loc[:, self.name] = self.start_date + data[self.name].cumsum(axis=0)
        else:
            data.loc[:, self.name] += self.min_date
        return data

    def features(self, x=None):
        features = super().features(x=x)
        features.update(self.hour.features(x=x))
        features.update(self.dow.features(x=x))
        features.update(self.day.features(x=x))
        features.update(self.month.features(x=x))
        return features

    @tensorflow_name_scoped
    def input_tensor(self, feed=None):
        delta = super().input_tensor(feed=feed)
        hour = self.hour.input_tensor(feed=feed)
        dow = self.dow.input_tensor(feed=feed)
        day = self.day.input_tensor(feed=feed)
        month = self.month.input_tensor(feed=feed)
        x = tf.concat(values=(delta, hour, dow, day, month), axis=1)
        return x

    # TODO: skip last and assume absolute value
    # def tf_loss(self, x, feed=None):

