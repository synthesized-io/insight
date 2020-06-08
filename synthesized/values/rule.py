from math import inf
from typing import Dict, Callable, Optional, List

import pandas as pd
import tensorflow as tf

from .value import Value
from ..common.module import tensorflow_name_scoped


class RuleValue(Value):

    def __init__(self, name: str, values: List[Value], function: str, fkwargs: Optional[Dict] = None):
        super().__init__(name=name)

        self.values = values
        self.functions: Dict[str, Callable] = dict()
        self.fkwargs = fkwargs
        if function == 'flag_1':
            assert all(len(value.columns()) == 1 for value in self.values)
            # TODO: seems we always assume fkwargs to be not None, why it's Option then?
            assert fkwargs is not None
            assert len(fkwargs['threshs']) + 1 == len(fkwargs['categories'])
            self.num_learned = 1

            def piecewise(x):
                y = pd.Series(self.fkwargs['categories'][0], index=x.index)
                y.loc[x.iloc[:, 0] < self.fkwargs['threshs'][0]] = self.fkwargs['categories'][0]
                regions = list(self.fkwargs['threshs']) + [inf]
                for i, (t1, t2) in enumerate(zip(regions[:-1], regions[1:])):
                    y.loc[(x.iloc[:, 0] >= t1) & (x.iloc[:, 0] < t2)] = self.fkwargs['categories'][i+1]
                return y
            self.functions[self.values[1].columns()[0]] = piecewise
        elif function == 'pulse_1':
            assert all(len(value.columns()) == 1 for value in self.values)
            # TODO: seems we always assume fkwargs to be not None, why it's Option then?
            assert fkwargs is not None
            assert len(fkwargs['threshs']) == len(fkwargs['categories']) == 2
            self.num_learned = 1

            def pulse(x):
                y = pd.Series(self.fkwargs['categories'][0], index=x.index)
                y.loc[(x.iloc[:, 0] > self.fkwargs['threshs']['lower']) &
                      (x.iloc[:, 0] < self.fkwargs['threshs']['upper'])] = self.fkwargs['categories'][1]
                return y
            self.functions[self.values[1].columns()[0]] = pulse
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
                #  df.loc[:, name] = df.apply(lambda row: self.functions[name](*row[columns]), axis=1)
                df.loc[:, name] = self.functions[name](df[columns])
        return df

    def module_initialize(self) -> None:
        raise NotImplementedError

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
    def output_tensors(self, y: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        splits = [value.learned_output_size() for value in self.values[:self.num_learned]]
        y = tf.split(value=y, num_or_size_splits=splits, axis=1)
        ys: List[tf.Tensor] = list()
        for value, y in zip(self.values[:self.num_learned], y):
            ys.extend(value.output_tensors(y=y, **kwargs))
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
