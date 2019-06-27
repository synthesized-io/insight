from typing import Dict, Callable
from typing import List

import pandas as pd
import tensorflow as tf

from .value import Value
from ..module import tensorflow_name_scoped


class RuleValue(Value):

    def __init__(self, name: str, values: List[Value], function: str):
        super().__init__(name=name)

        self.values = values

        self.functions: Dict[str, Callable] = dict()
        if function == 'pick-first':
            assert all(len(value.columns()) == 1 for value in self.values)
            self.num_learned = 2
            self.functions[self.values[2].columns()[0]] = lambda x, y: x
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__[:-5].lower()

    def columns(self) -> List[str]:
        columns = list()
        for value in self.values:
            columns.extend(value.columns())
        return columns

    def learned_input_columns(self) -> List[str]:
        columns = list()
        for value in self.values:
            columns.extend(value.learned_input_columns())
        return columns

    def learned_output_columns(self) -> List[str]:
        columns = list()
        for value in self.values[:self.num_learned]:
            columns.extend(value.learned_output_columns())
        return columns

    def learned_input_size(self) -> int:
        return sum(value.learned_input_size() for value in self.values)

    def learned_output_size(self) -> int:
        return sum(value.learned_output_size() for value in self.values[:self.num_learned])

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        for value in self.values:
            value.extract(df=df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for value in self.values:
            df = value.preprocess(df=df)
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = list()
        for value in self.values[:self.num_learned]:
            df = value.postprocess(df=df)
            columns.extend(value.columns())
        for value in self.values[self.num_learned:]:
            for name in value.columns():
                df.loc[:, name] = df.apply(lambda row: self.functions[name](*row[columns]), axis=1)
        return df

    def module_initialize(self) -> None:
        raise NotImplementedError

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        xs: List[tf.Tensor] = list()
        for value in self.values:
            xs.extend(value.input_tensors())
        return xs

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        x = list()
        index = 0
        for value in self.values:
            num = len(value.learned_input_columns())
            x.append(value.unify_inputs(xs=xs[index: index + num]))
            index += num
        return tf.concat(values=x, axis=1)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        splits = [value.learned_output_size() for value in self.values[:self.num_learned]]
        y = tf.split(value=y, num_or_size_splits=splits, axis=1)
        ys: List[tf.Tensor] = list()
        for value, y in zip(self.values[:self.num_learned], y):
            ys.extend(value.output_tensors(y=y))
        return ys

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        splits = [value.learned_output_size() for value in self.values[:self.num_learned]]
        y = tf.split(value=y, num_or_size_splits=splits, axis=1)
        losses = list()
        index = 0
        for value, y in zip(self.values[:self.num_learned], y):
            num = len(value.learned_output_columns())
            losses.append(value.loss(y=y, xs=xs[index: index + num]))
            index += num
        return tf.math.add_n(inputs=losses)

    @tensorflow_name_scoped
    def distribution_loss(self, samples: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
