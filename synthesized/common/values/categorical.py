import logging
from math import log
from typing import Any, Dict, Optional, Sequence

import numpy as np
import tensorflow as tf

from .value import Value
from ..module import tensorflow_name_scoped
from ..util import get_initializer
from ...config import CategoricalConfig

logger = logging.getLogger(__name__)


class CategoricalValue(Value):

    def __init__(
        self, name: str, num_categories: int,
        # Optional
        similarity_based: bool = False,
        # Scenario
        probabilities=None, embedding_size: int = None,
        config: CategoricalConfig = CategoricalConfig()
    ):
        super().__init__(name=name)
        self.num_categories: int = num_categories
        self.similarity_based = similarity_based
        self.probabilities = probabilities

        if embedding_size:
            self.embedding_size: Optional[int] = embedding_size
        else:
            self.embedding_size = compute_embedding_size(self.num_categories + 1, similarity_based=similarity_based)

        self.embedding_initialization = 'glorot-normal' if similarity_based else 'orthogonal-small'

        self.weight = config.categorical_weight
        self.use_moving_average = config.moving_average
        self.temperature = config.temperature

        self.moving_average: Optional[tf.train.ExponentialMovingAverage] = None
        self.embeddings: Optional[tf.Variable] = None
        self.frequency: Optional[tf.Variable] = None
        self.dtype = tf.int64

        self.build()

    def __str__(self) -> str:
        string = super().__str__()
        string += '{}-{}'.format(self.num_categories, self.embedding_size)
        if self.similarity_based:
            string += '-similarity'
        return string

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            embedding_size=self.embedding_size,
            similarity_based=self.similarity_based,
            weight=self.weight, temperature=self.temperature, moving_average=self.use_moving_average,
            embedding_initialization=self.embedding_initialization
        )
        return spec

    def learned_input_size(self) -> int:
        assert self.embedding_size is not None
        return self.embedding_size

    def learned_output_size(self) -> int:
        assert self.num_categories is not None
        return self.num_categories + 1

    @tensorflow_name_scoped
    def build(self):
        if not self.built:
            if self.probabilities is not None and not self.similarity_based:
                # "hack": scenario synthesizer, embeddings not used
                return
            shape = (self.learned_output_size(), self.embedding_size)
            initializer = get_initializer(initializer=self.embedding_initialization)

            self.embeddings = tf.Variable(
                initial_value=initializer(shape=shape, dtype=tf.float32), name='embeddings', shape=shape,
                dtype=tf.float32, trainable=True, caching_device=None, validate_shape=True
            )
            self.add_regularization_weight(self.embeddings)

            if self.use_moving_average:
                self.moving_average = tf.train.ExponentialMovingAverage(decay=0.9)
                self.frequency = tf.Variable(
                    initial_value=np.zeros(shape=(self.learned_output_size(),)), trainable=False, dtype=tf.float32,
                    name='moving_avg_freq'
                )
                self.moving_average.apply(var_list=[self.frequency])

        self.built = True

    @tensorflow_name_scoped
    def unify_inputs(self, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        self.build()
        return tf.nn.embedding_lookup(params=self.embeddings, ids=xs[0])

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, sample: bool = True, **kwargs) -> Sequence[tf.Tensor]:

        # Choose argmax class
        y_flat = tf.reshape(y, shape=(-1, y.shape[-1]))

        if sample:
            y_flat = tf.random.categorical(logits=y_flat[:, 1:], num_samples=1) + 1
        else:
            y_flat = tf.expand_dims(tf.argmax(y_flat[:, 1:], axis=1) + 1, axis=1)

        y = tf.reshape(y_flat, shape=tf.concat(([-1], y.shape[1:-1]), axis=0))

        return (y,)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        target = xs[0]
        # mask loss from nan inputs
        mask = tf.math.logical_not(target == 0)

        if self.moving_average is not None:
            assert self.frequency is not None
            flattened_target = tf.reshape(target, shape=(-1,))
            frequency = tf.concat(values=(np.array(range(self.learned_output_size())), flattened_target), axis=0)
            _, _, frequency = tf.unique_with_counts(x=frequency, out_idx=tf.int32)
            frequency = tf.reshape(tensor=frequency, shape=(self.learned_output_size(),))
            frequency = tf.cast(frequency - 1, dtype=tf.float32)
            frequency = frequency / tf.reduce_sum(input_tensor=frequency, axis=0, keepdims=False)
            with tf.control_dependencies([self.frequency.assign(frequency)]):
                self.moving_average.apply(var_list=[self.frequency, ])
                frequency = self.moving_average.average(var=self.frequency)
                frequency = tf.nn.embedding_lookup(
                    params=frequency, ids=flattened_target, max_norm=None,
                    name='frequency'
                )
                weights = tf.sqrt(x=(1.0 / tf.maximum(x=frequency, y=1e-6)))
                weights = tf.dtypes.cast(x=weights, dtype=tf.float32)
                weights = tf.reshape(weights, shape=target.shape)
                # weights = 1.0 / tf.maximum(x=frequency, y=1e-6)
                weights = tf.boolean_mask(tensor=weights, mask=mask)
        else:
            weights = 1.0

        y = y / self.temperature

        target = tf.one_hot(
            indices=target, depth=self.learned_output_size(), on_value=1.0, off_value=0.0, axis=-1,
            dtype=tf.float32
        )

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y, axis=-1)
        loss = tf.boolean_mask(loss, mask=mask)
        loss = self.weight * tf.reduce_mean(input_tensor=(loss * weights), axis=None)

        tf.summary.scalar(name=self.name, data=loss)
        return loss

    def distribution_loss(self, y: tf.Tensor) -> tf.Tensor:

        if self.probabilities is None:
            return tf.constant(value=0.0, dtype=tf.float32)

        samples = y
        num_samples = tf.shape(input=samples)[0]
        samples = tf.concat(
            values=(tf.range(start=0, limit=self.learned_output_size(), dtype=tf.int64), samples), axis=0
        )
        _, _, counts = tf.unique_with_counts(x=samples)
        counts = counts - 1
        probs = tf.cast(x=counts, dtype=tf.float32) / tf.cast(x=num_samples, dtype=tf.float32)
        loss = tf.math.squared_difference(x=probs, y=self.probabilities)
        loss = tf.reduce_mean(input_tensor=loss, axis=0)
        return loss


def compute_embedding_size(num_categories: Optional[int], similarity_based: bool = False) -> Optional[int]:
    if similarity_based and num_categories:
        return int(log(num_categories + 1) * 2.0)
    else:
        return num_categories
