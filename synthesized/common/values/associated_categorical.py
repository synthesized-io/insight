import logging
from typing import Any, Dict, List

import pandas as pd
import tensorflow as tf

from .categorical import CategoricalValue
from .value import Value
from ..module import tensorflow_name_scoped

logger = logging.getLogger(__name__)


class AssociatedCategoricalValue(Value):
    def __init__(
            self, values: List[CategoricalValue]
    ):
        super(AssociatedCategoricalValue, self).__init__(
            name='|'.join([v.name for v in values])
        )
        self.values = values
        self.dtype = tf.int64

    def __str__(self) -> str:
        string = super().__str__()
        return string

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            categories=self.categories, embedding_size=self.embedding_size,
            similarity_based=self.similarity_based,
            weight=self.weight, temperature=self.temperature, moving_average=self.use_moving_average,
            produce_nans=self.produce_nans, embedding_initialization=self.embedding_initialization
        )
        return spec

    def columns(self) -> List[str]:
        return [name for value in self.values for name in value.columns()]

    def learned_input_columns(self) -> List[str]:
        return [name for value in self.values for name in value.learned_input_columns()]

    def learned_output_columns(self) -> List[str]:
        return [name for value in self.values for name in value.learned_output_columns()]

    def learned_input_size(self) -> int:
        return sum([v.learned_input_size() for v in self.values])

    def learned_output_size(self) -> int:
        return sum([v.learned_output_size() for v in self.values])

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        for v in self.values:
            v.extract(df=df)

        self.build()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for value in self.values:
            df = value.preprocess(df)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        for value in self.values:
            df = value.postprocess(df)
        return df

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        return tf.concat([value.unify_inputs(xs[n:n+1]) for n, value in enumerate(self.values)], axis=-1)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        # TODO: Correctly output categories that are bound to each other.
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=-1
        )
        return [t for n, value in enumerate(self.values) for t in value.output_tensors(ys[n])]

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=-1
        )
        return tf.reduce_sum([v.loss(y=ys[n], xs=xs[n:n+1]) for n, v in enumerate(self.values)], axis=None)
