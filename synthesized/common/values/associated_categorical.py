import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import tensorflow as tf

from .categorical import CategoricalValue
from .value import Value
from synthesized.common.module import tensorflow_name_scoped

logger = logging.getLogger(__name__)


class AssociatedCategoricalValue(Value):
    def __init__(
            self, values: List[CategoricalValue], associations: List[List[str]]
    ):
        super(AssociatedCategoricalValue, self).__init__(
            name='|'.join([v.name for v in values])
        )
        self.values = values
        logger.debug("Creating Associated value with associations: ")
        for n, association in enumerate(associations):
            logger.debug(f"{n+1}: {association}")
        self.associations = associations
        self.dtype = tf.int64
        self.binding_mask: Optional[tf.Tensor] = None

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

        final_mask = np.ones(shape=[v.num_categories for v in self.values])

        for associated_values in self.associations:
            associated_values = list(associated_values)
            associated_values.sort(key=lambda x: [v.name for v in self.values].index(x))
            df2 = df[associated_values].copy()
            for v in self.values:
                df2[v.name] = df2[v.name].map(v.category2idx)

            counts = np.zeros(shape=[df2[v].nunique() for v in associated_values])

            for i, row in df2.iterrows():
                idx = tuple(v for v in row.values)
                counts[idx] += 1

            mask = (counts > 0).astype(np.int32)

            for n, v in enumerate(self.values):
                if v.name not in associated_values:
                    mask = np.expand_dims(mask, axis=n)

            final_mask *= np.broadcast_to(mask, [v.num_categories for v in self.values])

        self.binding_mask = tf.constant(final_mask, dtype=tf.float32)

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
        """Outputs the bound categorical values."""
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=-1
        )

        probs = []
        for y in ys:
            y_flat = tf.reshape(y, shape=(-1, y.shape[-1]))
            prob = tf.math.softmax(y_flat, axis=-1)
            probs.append(prob)

        joint = tf_joint_probs(*probs)
        masked = tf_masked_probs(joint, self.binding_mask)
        flattened = tf.reshape(masked, (masked.shape[0], -1))

        y = tf.reshape(tf.random.categorical(tf.math.log(flattened), num_samples=1), shape=flattened.shape[0:-1])
        ot = [tf.math.mod(y, self.values[-1].num_categories)]
        for n in range(1, len(self.values)):
            ot.append(tf.math.mod(
                tf.math.floordiv(
                    y,
                    tf.cast(tf.reduce_prod([self.values[-m-1].num_categories for m in range(n)]), dtype=tf.int64)
                ),
                self.values[-n-1].num_categories
            ))

        return ot[::-1]

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=-1
        )
        return tf.reduce_sum([v.loss(y=ys[n], xs=xs[n:n+1]) for n, v in enumerate(self.values)], axis=None)


def tf_joint_probs(*args):
    rank = len(args)
    probs = []
    for n, x in enumerate(args):
        for m in range(n):
            x = tf.expand_dims(x, axis=-2-m)
        for m in range(rank-n-1):
            x = tf.expand_dims(x, axis=-1)
        probs.append(x)

    joint_prob = probs[0]
    for n in range(1, rank):
        joint_prob = joint_prob * probs[n]

    return joint_prob


def tf_masked_probs(jp, mask):
    if jp.shape[-len(mask.shape):] != mask.shape:
        raise ValueError("Mask shape doesn't match joint probability's shape.")

    d = jp * mask
    for n in range(len(jp.shape)-len(mask.shape), len(jp.shape)):
        d_a = (tf.reduce_sum(jp, axis=n, keepdims=True) / tf.reduce_sum(jp * mask, axis=n, keepdims=True) - 1) * (
                jp * mask)
        d += d_a

    return d
