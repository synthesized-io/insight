from math import isnan, log
from typing import Any, Dict, List, Optional
import logging

import pandas as pd
import tensorflow as tf
import numpy as np

from .value import Value
from .. import util
from ..module import tensorflow_name_scoped

logger = logging.getLogger(__name__)


class CategoricalValue(Value):

    def __init__(
        self, name: str, capacity: int, weight: float, temperature: float,
        moving_average: bool,
        # Optional
        similarity_based: bool = False, pandas_category: bool = False, produce_nans: bool = False,
        # Scenario
        categories: List = None, probabilities=None, embedding_size: int = None
    ):
        super().__init__(name=name)
        self.categories: Optional[List] = None
        self.category2idx: Optional[Dict] = None
        self.idx2category: Optional[Dict] = None
        self.nans_valid: bool = False
        if categories is None:
            self.num_categories: Optional[int] = None
        else:
            unique_values = pd.Series(categories).unique().tolist()
            self._set_categories(unique_values)

        self.probabilities = probabilities
        self.capacity = capacity

        if embedding_size:
            self.embedding_size: Optional[int] = embedding_size
        else:
            self.embedding_size = compute_embedding_size(self.num_categories, similarity_based=similarity_based)

        if similarity_based:
            self.embedding_initialization = 'glorot-normal'
        else:
            self.embedding_initialization = 'orthogonal-small'

        self.weight = weight

        self.use_moving_average: bool = moving_average
        self.moving_average: Optional[tf.train.ExponentialMovingAverage] = None

        self.embeddings: Optional[tf.Variable] = None
        self.frequency: Optional[tf.Variable] = None
        self.similarity_based = similarity_based
        self.temperature = temperature
        self.pandas_category = pandas_category
        self.produce_nans = produce_nans
        self.is_string = False
        self.dtype = tf.int64

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
            similarity_based=self.similarity_based,
            weight=self.weight, temperature=self.temperature, moving_average=self.use_moving_average,
            produce_nans=self.produce_nans, embedding_initialization=self.embedding_initialization
        )
        return spec

    def learned_input_size(self) -> int:
        assert self.embedding_size is not None
        return self.embedding_size

    def learned_output_size(self) -> int:
        assert self.num_categories is not None
        return self.num_categories

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        if df.loc[:, self.name].dtype.kind == 'O':
            self.is_string = True
            df.loc[:, self.name].fillna('nan', inplace=True)

        unique_values = df.loc[:, self.name].unique().tolist()
        self._set_categories(unique_values)

        if self.embedding_size is None:
            self.embedding_size = compute_embedding_size(self.num_categories, similarity_based=self.similarity_based)

        self.build()

    @tensorflow_name_scoped
    def build(self):
        if not self.built:
            if self.probabilities is not None and not self.similarity_based:
                # "hack": scenario synthesizer, embeddings not used
                return
            shape = (self.num_categories, self.embedding_size)
            initializer = util.get_initializer(initializer=self.embedding_initialization)

            self.embeddings = tf.Variable(
                initial_value=initializer(shape=shape, dtype=tf.float32), name='embeddings', shape=shape,
                dtype=tf.float32, trainable=True, caching_device=None, validate_shape=True
            )
            self.add_regularization_weight(self.embeddings)

            if self.use_moving_average:
                self.moving_average = tf.train.ExponentialMovingAverage(decay=0.9)
                self.frequency = tf.Variable(
                    initial_value=np.zeros(shape=(self.num_categories,)), trainable=False, dtype=tf.float32,
                    name='moving_avg_freq'
                )
                self.moving_average.apply(var_list=[self.frequency])

        self.built = True

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_string:
            df.loc[:, self.name] = df.loc[:, self.name].fillna('nan')

        assert isinstance(self.categories, list)
        df.loc[:, self.name] = df.loc[:, self.name].map(self.category2idx)

        if df.loc[:, self.name].dtype != 'int64':
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='int64')
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        assert isinstance(self.categories, list)
        df.loc[:, self.name] = df.loc[:, self.name].map(self.idx2category)
        if self.is_string:
            df.loc[df[self.name] == 'nan', self.name] = np.nan

        if self.pandas_category:
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='category')
        return df

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        self.build()
        return tf.nn.embedding_lookup(params=self.embeddings, ids=xs[0])

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        if self.nans_valid is True and self.produce_nans is False and self.num_categories == 1:
            logger.warning("CategoricalValue '{}' is set to produce nans, but a single nan category has been learned. "
                           "Setting 'procude_nans=True' for this column".format(self.name))
            self.produce_nans = True

        # Choose argmax class
        if self.nans_valid is False or self.produce_nans:
            y = tf.squeeze(tf.random.categorical(logits=y, num_samples=1, dtype=tf.int64))
        else:
            # If we don't want to produce nans, the argmax won't consider the probability of class 0 (nan).
            y = tf.squeeze(tf.random.categorical(logits=y[:, 1:], num_samples=1, dtype=tf.int64)) + 1

        return [y]

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        target = xs[0]
        if self.moving_average is not None:
            assert self.num_categories is not None
            assert self.frequency is not None
            frequency = tf.concat(values=(np.array(range(self.num_categories)), target), axis=0)
            _, _, frequency = tf.unique_with_counts(x=frequency, out_idx=tf.int32)
            frequency = tf.reshape(tensor=frequency, shape=(self.num_categories,))
            frequency = tf.cast(frequency - 1, dtype=tf.float32)
            frequency = frequency / tf.reduce_sum(input_tensor=frequency, axis=0, keepdims=False)
            with tf.control_dependencies([self.frequency.assign(frequency)]):
                self.moving_average.apply(var_list=[self.frequency, ])
                frequency = self.moving_average.average(var=self.frequency)
                frequency = tf.nn.embedding_lookup(
                    params=frequency, ids=target, max_norm=None,
                    name='frequency'
                )
                weights = tf.sqrt(x=(1.0 / tf.maximum(x=frequency, y=1e-6)))
                weights = tf.dtypes.cast(x=weights, dtype=tf.float32)
                # weights = 1.0 / tf.maximum(x=frequency, y=1e-6)
        else:
            weights = 1.0

        y = y / self.temperature
        assert self.num_categories is not None

        target = tf.one_hot(
            indices=xs[0], depth=self.num_categories, on_value=1.0, off_value=0.0, axis=1,
            dtype=tf.float32
        )
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y, axis=1)
        loss = self.weight * tf.reduce_mean(input_tensor=(loss * weights), axis=0)
        tf.summary.scalar(name=self.name, data=loss)
        return loss

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
        loss = tf.math.squared_difference(x=probs, y=self.probabilities)
        loss = tf.reduce_mean(input_tensor=loss, axis=0)
        return loss

    def _set_categories(self, categories: list):

        found = None
        # Put any nan at the position zero of the list
        for n, x in enumerate(categories):
            if isinstance(x, float) and isnan(x):
                found = categories.pop(n)
                self.nans_valid = True
                break

        categories = np.sort(categories).tolist()
        if found is not None:
            if self.is_string:
                categories.insert(0, 'nan')
            else:
                categories.insert(0, np.nan)

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


def compute_embedding_size(num_categories: Optional[int], similarity_based: bool = False) -> Optional[int]:
    if similarity_based and num_categories:
        return int(log(num_categories + 1) * 2.0)
    else:
        return num_categories
