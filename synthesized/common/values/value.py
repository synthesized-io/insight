from typing import List

import pandas as pd
import tensorflow as tf

from ..module import Module, tensorflow_name_scoped


class Value(Module):

    def __init__(self, name: str):
        super().__init__(name=name)

    def __str__(self) -> str:
        return self.__class__.__name__[:-5].lower()

    def columns(self) -> List[str]:
        return [self.name]

    def learned_input_columns(self) -> List[str]:
        if self.learned_input_size() == 0:
            return list()
        else:
            return [self.name]

    def learned_output_columns(self) -> List[str]:
        if self.learned_output_size() == 0:
            return list()
        else:
            return [self.name]

    def learned_input_size(self) -> int:
        return 0

    def learned_output_size(self) -> int:
        return 0

    def extract(self, df: pd.DataFrame) -> None:
        # begin
        assert all(name in df.columns for name in self.columns())

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # end
        assert all(name in df.columns for name in self.learned_input_columns())
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # begin
        assert all(name in df.columns for name in self.learned_output_columns())
        return df

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        return list()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        raise NotImplementedError

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        return list()

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        return tf.constant(value=0.0, dtype=tf.float32)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        return tf.constant(value=0.0, dtype=tf.float32)
