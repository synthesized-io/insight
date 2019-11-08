from math import isnan, log
from typing import Any, Dict, List, Union, Optional

import pandas as pd
import tensorflow as tf

from .value import Value
from .. import util
from ..module import tensorflow_name_scoped


class CategoricalValue(Value):

    def __init__(
        self, name: str, capacity: int, weight_decay: float, weight: float, temperature: float,
        smoothing: float, moving_average: tf.train.ExponentialMovingAverage, similarity_regularization: float,
        entropy_regularization: float,
        # Optional
        similarity_based: bool = False, pandas_category: bool = False,
        # Scenario
        categories: Union[int, list] = None, probabilities=None, embedding_size: int = None
    ):
        super().__init__(name=name)
        self.categories: Optional[Union[int, list]] = None
        self.category2idx: Optional[Dict] = None
        self.idx2category: Optional[Dict] = None
        self.nans_valid: bool = False
        if categories is None:
            self.num_categories = None
        elif isinstance(categories, int):
            self.categories = self.num_categories = categories
        else:
            unique_values = list(pd.Series(categories).unique())
            self._set_categories(unique_values)

        self.probabilities = probabilities

        self.capacity = capacity
        if embedding_size is None and self.num_categories is not None:
            embedding_size = compute_embedding_size(self.num_categories, self.capacity)
        self.embedding_size = embedding_size
        self.weight_decay = weight_decay
        self.weight = weight

        self.smoothing = smoothing
        self.moving_average: Optional[tf.train.ExponentialMovingAverage] = moving_average
        self.similarity_regularization = similarity_regularization
        self.entropy_regularization = entropy_regularization

        self.similarity_based = similarity_based
        self.temperature = temperature
        self.pandas_category = pandas_category

    def __str__(self) -> str:
        string = super().__str__()
        string += '{}-{}'.format(self.num_categories, self.embedding_size)
        if self.similarity_based:
            string += '-similarity'
        return string

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            categories=self.categories, embedding_size=self.embedding_size,
            similarity_based=self.similarity_based, weight_decay=self.weight_decay,
            weight=self.weight, temperature=self.temperature, smoothing=self.smoothing,
            moving_average=self.moving_average,
            similarity_regularization=self.similarity_regularization,
            entropy_regularization=self.entropy_regularization
        )
        return spec

    def learned_input_size(self) -> int:
        assert self.embedding_size is not None
        return self.embedding_size

    def learned_output_size(self) -> int:
        if self.similarity_based:
            assert self.embedding_size is not None
            return self.embedding_size
        else:
            assert self.num_categories is not None
            return self.num_categories

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        unique_values = list(df[self.name].unique())
        self._set_categories(unique_values)

        if self.embedding_size is None and self.num_categories is not None:
            self.embedding_size = compute_embedding_size(self.num_categories, self.capacity)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(self.categories, int):
            assert isinstance(self.categories, list)
            df[self.name] = df[self.name].map(self.category2idx)
        if df[self.name].dtype != 'int64':
            df[self.name] = df[self.name].astype(dtype='int64')
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        if not isinstance(self.categories, int):
            assert isinstance(self.categories, list)
            df[self.name] = df[self.name].map(self.idx2category)
        if self.pandas_category:
            df[self.name] = df[self.name].astype(dtype='category')
        return df

    def module_initialize(self) -> None:
        super().module_initialize()

        # Input placeholder for value
        self.placeholder_initialize(dtype=tf.int64, shape=(None,))

        if self.probabilities is not None and not self.similarity_based:
            # "hack": scenario synthesizer, embeddings not used
            return
        shape = (self.num_categories, self.embedding_size)
        initializer = util.get_initializer(initializer='normal')
        regularizer = util.get_regularizer(regularizer='l2', weight=self.weight_decay)
        self.embeddings = tf.compat.v1.get_variable(
            name='embeddings', shape=shape, dtype=tf.float32, initializer=initializer,
            regularizer=regularizer, trainable=True
        )
        if self.moving_average:
            self.moving_average = tf.train.ExponentialMovingAverage(decay=0.9)
        else:
            self.moving_average = None

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        return [self.placeholder]

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 1
        return tf.nn.embedding_lookup(params=self.embeddings, ids=xs[0])

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        if self.similarity_based:
            # Similarities as logits
            y = tf.expand_dims(input=y, axis=1)
            embeddings = tf.expand_dims(input=self.embeddings, axis=0)
            y = tf.reduce_sum(input_tensor=(y * embeddings), axis=2, keepdims=False)

        # Choose argmax class
        y = tf.argmax(input=y, axis=1)

        return [y]

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 1
        target = xs[0]

        if self.moving_average is not None:
            assert self.num_categories is not None
            frequency = tf.concat(values=(list(range(self.num_categories)), target), axis=0)
            _, _, frequency = tf.unique_with_counts(x=frequency, out_idx=tf.int32)
            frequency = tf.reshape(tensor=frequency, shape=(self.num_categories,))
            frequency = frequency - 1
            frequency = frequency / tf.reduce_sum(input_tensor=frequency, axis=0, keepdims=False)
            update = self.moving_average.apply(var_list=(frequency,))
            with tf.control_dependencies(control_inputs=(update,)):
                frequency = self.moving_average.average(var=frequency)
                frequency = tf.nn.embedding_lookup(
                    params=frequency, ids=target, partition_strategy='mod', validate_indices=True,
                    max_norm=None
                )
                weights = tf.sqrt(x=(1.0 / tf.maximum(x=frequency, y=1e-6)))
                weights = tf.dtypes.cast(x=weights, dtype=tf.float32)
                # weights = 1.0 / tf.maximum(x=frequency, y=1e-6)
        else:
            weights = 1.0
        target = tf.one_hot(
            indices=target, depth=self.num_categories, on_value=1.0, off_value=0.0, axis=1,
            dtype=tf.float32
        )
        if self.similarity_based:  # is that right?
            y = tf.expand_dims(input=y, axis=1)
            embeddings = tf.expand_dims(input=self.embeddings, axis=0)
            y = tf.reduce_sum(input_tensor=(y * embeddings), axis=2, keepdims=False)
        y = y / self.temperature
        assert self.num_categories is not None
        target = target * (1.0 - self.smoothing) + self.smoothing / self.num_categories
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=y, axis=1)
        # loss = tf.squeeze(input=tf.math.squared_difference(x=y, y=target), axis=1)
        loss = self.weight * tf.reduce_mean(input_tensor=(loss * weights), axis=0)
        if self.similarity_regularization > 0.0:
            similarity_loss = tf.matmul(a=self.embeddings, b=self.embeddings, transpose_b=True)
            similarity_loss = tf.reduce_sum(input_tensor=similarity_loss, axis=1)
            similarity_loss = tf.reduce_sum(input_tensor=similarity_loss, axis=0)
            similarity_loss = self.similarity_regularization * similarity_loss
            loss = loss + similarity_loss
        if self.entropy_regularization > 0.0:
            probs = tf.nn.softmax(logits=y, axis=-1)
            logprobs = tf.math.log(x=tf.maximum(x=probs, y=1e-6))
            entropy_loss = -tf.reduce_sum(input_tensor=(probs * logprobs), axis=1)
            entropy_loss = tf.reduce_sum(input_tensor=entropy_loss, axis=0)
            entropy_loss *= -self.entropy_regularization
            loss = loss + entropy_loss
        return loss

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        assert len(ys) == 1

        if self.probabilities is None:
            return tf.constant(value=0.0, dtype=tf.float32)

        samples = ys[0]
        num_samples = tf.shape(input=samples)[0]
        samples = tf.concat(
            values=(tf.range(start=0, limit=self.num_categories, dtype=tf.int64), samples), axis=0
        )
        _, _, counts = tf.unique_with_counts(x=samples)
        counts = counts - 1
        probs = tf.cast(x=counts, dtype=tf.float32) / tf.cast(x=num_samples, dtype=tf.float32)
        loss = tf.squared_difference(x=probs, y=self.probabilities)
        loss = tf.reduce_mean(input_tensor=loss, axis=0)
        return loss

    def _set_categories(self, categories: list):

        # Put any nan at the position zero of the list
        for n, x in enumerate(categories):
            if isinstance(x, float) and isnan(x):
                categories.insert(0, categories.pop(n))
                self.nans_valid = True
            else:
                assert not isinstance(x, float) or not isnan(x)

        # If categories are not set
        if self.categories is None:
            self.categories = categories
            self.num_categories = len(self.categories)
            self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
            self.idx2category = {i: self.categories[i] for i in range(len(self.categories))}
        # If categories have been set and are different to the given
        elif isinstance(self.categories, list) and \
                categories[int(self.nans_valid):] != self.categories[int(self.nans_valid):]:
            raise NotImplementedError


def compute_embedding_size(num_categories: int, capacity: int) -> int:
    return max(int(log(num_categories + 1) * capacity / 8.0), capacity)
