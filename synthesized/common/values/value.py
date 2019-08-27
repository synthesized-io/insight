from typing import List, Optional, Tuple

import pandas as pd
import tensorflow as tf

from ..module import Module, tensorflow_name_scoped


class Value(Module):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.name = name
        self.placeholder: Optional[tf.Tensor] = None  # not all values have a placeholder

    def placeholder_initialize(self, dtype: tf.DType, shape: Tuple):
        assert self.placeholder is None
        self.placeholder = tf.placeholder(
            dtype=dtype, shape=shape, name=self.make_tf_compatible(string=self.name)
        )

    def __str__(self) -> str:
        return self.__class__.__name__[:-5].lower()

    def columns(self) -> List[str]:
        """External columns which are covered by this value.

        Returns:
            Columns covered by this value.

        """
        return [self.name]

    def learned_input_columns(self) -> List[str]:
        """Internal input columns for a generative model.

        Returns:
            Learned input columns.

        """
        if self.learned_input_size() == 0:
            return list()
        else:
            return [self.name]

    def learned_output_columns(self) -> List[str]:
        """Internal output columns for a generative model.

        Returns:
            Learned output columns.

        """
        if self.learned_output_size() == 0:
            return list()
        else:
            return [self.name]

    def learned_input_size(self) -> int:
        """Internal input embedding size for a generative model.

        Returns:
            Learned input embedding size.

        """
        return 0

    def learned_output_size(self) -> int:
        """Internal output embedding size for a generative model.

        Returns:
            Learned output embedding size.

        """
        return 0

    def extract(self, df: pd.DataFrame) -> None:
        """Extracts configuration parameters from a representative data frame.

        Overwriting implementations should call super().extract(df=df) as first step.

        Args:
            df: Representative data frame.

        """
        assert all(name in df.columns for name in self.columns())

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-processes a data frame to prepare it as input for a generative model. This may
        include adding or removing columns in case of `learned_input_columns()` differing from
        `columns()`.

        Important: this function modifies the given data frame.

        Overwriting implementations should call super().preprocess(df=df) as last step.

        Args:
            df: Data frame to be pre-processed.

        Returns:
            Pre-processed data frame.

        """
        assert all(name in df.columns for name in self.learned_input_columns())
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes a data frame, usually the output of a generative model. Post-processing
        basically reverses the pre-processing procedure. This may include re-introducing columns in
        case of `learned_output_columns()` differing from `columns()`.

        Important: this function modifies the given data frame.

        Overwriting implementations should call super().postprocess(df=df) as first step.

        Args:
            df: Data frame to be post-processed.

        Returns:
            Post-processed data frame.

        """
        assert all(name in df.columns for name in self.learned_output_columns())
        return df

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        """Input tensors.

        Returns:
            Input tensors, one per `learned_input_columns()`.

        """
        return list()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        """Unifies input tensors into a single input embedding for a generative model.

        Args:
            xs: Input tensors, one per `learned_input_columns()`, usually from `input_tensors()`.

        Returns:
            Input embedding, of size `learned_input_size()`.

        """
        raise NotImplementedError

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        """Turns an output embedding of a generative model into corresponding output tensors.

        Args:
            y: Output embedding, of size `learned_output_size()`.

        Returns:
            Output tensors, one per `learned_output_columns()`.

        """
        return list()

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        """Computes the reconstruction loss of an output embedding and corresponding input tensors.

        Args:
            y: Output embedding, of size `learned_output_size()`.
            xs: Input tensors, one per `learned_input_columns()`, usually from `input_tensors()`.

        Returns:
            Loss.

        """
        return tf.constant(value=0.0, dtype=tf.float32)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        """Computes the distributional distance of sample output embeddings to the target
        distribution.

        Args:
            ys: Sample output embeddings, of size `learned_output_size()`.

        Returns:
            Loss.

        """
        return tf.constant(value=0.0, dtype=tf.float32)
