from typing import Dict, Iterable

import pandas as pd
import tensorflow as tf

from ..module import Module, tensorflow_name_scoped


class Value(Module):

    def __init__(self, name: str):
        super().__init__(name=name)

    def __str__(self) -> str:
        return self.__class__.__name__[:-5].lower()

    def input_size(self) -> int:
        return 0

    def output_size(self) -> int:
        return 0

    def input_labels(self) -> Iterable[str]:
        if self.input_size() > 0:
            yield self.name

    def output_labels(self) -> Iterable[str]:
        if self.output_size() > 0:
            yield self.name

    def placeholders(self) -> Iterable[tf.Tensor]:
        return
        yield

    def extract(self, data: pd.DataFrame) -> None:
        pass

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def postprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def features(self, x=None):
        return dict()

    @tensorflow_name_scoped
    def input_tensor(self, feed: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        return None

    @tensorflow_name_scoped
    def output_tensors(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        return dict()

    @tensorflow_name_scoped
    def loss(self, x: tf.Tensor, feed: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        return None

    @tensorflow_name_scoped
    def distribution_loss(self, samples: tf.Tensor) -> tf.Tensor:
        return None
