from math import log
import tensorflow as tf

from .value import Value
from .. import util
from ..module import Module


class CategoricalValue(Value):

    def __init__(
        self, name, categories=None, capacity=None, embedding_size=None, pandas_category=False,
        similarity_based=False, weight_decay=0.0, temperature=1.0, smoothing=0.1,
        moving_average=True, similarity_regularization=0.1, entropy_regularization=0.1
    ):
        super().__init__(name=name)

        if categories is None:
            self.categories = None
            self.num_categories = None
        elif isinstance(categories, int):
            self.categories = self.num_categories = categories
        else:
            self.categories = sorted(categories)
            self.num_categories = len(self.categories)

        self.capacity = capacity
        if embedding_size is None and self.num_categories is not None:
            self.embedding_size = int(log(self.num_categories) * self.capacity / 2.0)
        else:
            self.embedding_size = embedding_size

        self.pandas_category = pandas_category
        self.similarity_based = similarity_based
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.smoothing = smoothing
        self.moving_average = moving_average
        self.similarity_regularization = similarity_regularization
        self.entropy_regularization = entropy_regularization

    def __str__(self):
        string = super().__str__()
        string += '{}-{}'.format(self.num_categories, self.embedding_size)
        if self.similarity_based:
            string += '-similarity'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            categories=self.categories, embedding_size=self.embedding_size,
            similarity_based=self.similarity_based, weight_decay=self.weight_decay,
            temperature=self.temperature, smoothing=self.smoothing,
            moving_average=self.moving_average,
            similarity_regularization=self.similarity_regularization,
            entropy_regularization=self.entropy_regularization
        )
        return spec

    def input_size(self):
        return self.embedding_size

    def output_size(self):
        if self.similarity_based:
            return self.embedding_size
        else:
            return self.num_categories

    def trainable_labels(self):
        yield self.name

    def placeholders(self):
        yield self.placeholder

    def extract(self, data):
        if self.categories is None:
            self.categories = sorted(data[self.name].unique())
            self.num_categories = len(self.categories)
            if self.embedding_size is None:
                self.embedding_size = int(log(self.num_categories) * self.capacity / 2.0)
        elif sorted(data[self.name].unique()) != self.categories:
            raise NotImplementedError

    def preprocess(self, data):
        if not isinstance(self.categories, int):
            data.loc[:, self.name] = data[self.name].map(arg=self.categories.index)
        data.loc[:, self.name] = data[self.name].astype(dtype='int64')
        return data

    def postprocess(self, data):
        if not isinstance(self.categories, int):
            data.loc[:, self.name] = data[self.name].map(arg=self.categories.__getitem__)
        if self.pandas_category:
            data.loc[:, self.name] = data[self.name].astype(dtype='category')
        return data

    def feature(self, x=None):
        if x is None:
            return tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=(x,)))

    def tf_initialize(self):
        super().tf_initialize()
        self.placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='input')
        assert self.name not in Module.placeholders
        Module.placeholders[self.name] = self.placeholder
        shape = (self.num_categories, self.embedding_size)
        initializer = util.get_initializer(initializer='normal')
        regularizer = util.get_regularizer(regularizer='l2', weight=self.weight_decay)
        self.embeddings = tf.get_variable(
            name='embeddings', shape=shape, dtype=tf.float32, initializer=initializer,
            regularizer=regularizer, trainable=True, collections=None, caching_device=None,
            partitioner=None, validate_shape=True, use_resource=None, custom_getter=None
        )
        if self.moving_average:
            self.moving_average = tf.train.ExponentialMovingAverage(
                decay=0.9, num_updates=None, zero_debias=False
            )
        else:
            self.moving_average = None

    def tf_input_tensor(self, feed=None):
        # tensor = tf.one_hot(
        #     indices=self.placeholder, depth=self.num_categories, on_value=1.0, off_value=0.0,
        #     axis=1, dtype=tf.float32
        # )
        x = self.placeholder if feed is None else feed
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=x, partition_strategy='mod', validate_indices=True,
            max_norm=None
        )
        return x

    def tf_output_tensors(self, x):
        if self.similarity_based:
            x = tf.expand_dims(input=x, axis=1)
            embeddings = tf.expand_dims(input=self.embeddings, axis=0)
            x = tf.reduce_sum(input_tensor=(x * embeddings), axis=2, keepdims=False)
        x = tf.argmax(input=x, axis=1)
        return {self.name: x}

    def tf_loss(self, x, feed=None):
        target = self.placeholder if feed is None else feed
        if self.moving_average is not None:
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
                # weights = 1.0 / tf.maximum(x=frequency, y=1e-6)
        else:
            weights = 1.0
        target = tf.one_hot(
            indices=target, depth=self.num_categories, on_value=1.0, off_value=0.0, axis=1,
            dtype=tf.float32
        )
        if self.similarity_based:  # is that right?
            x = tf.expand_dims(input=x, axis=1)
            embeddings = tf.expand_dims(input=self.embeddings, axis=0)
            x = tf.reduce_sum(input_tensor=(x * embeddings), axis=2, keepdims=False)
        x = x / self.temperature
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target, logits=x, weights=weights, label_smoothing=self.smoothing,
            scope=None, loss_collection=tf.GraphKeys.LOSSES
        )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        if self.similarity_regularization > 0.0:
            similarity_loss = tf.matmul(
                a=self.embeddings, b=self.embeddings, transpose_a=False, transpose_b=True,
                adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False
            )
            similarity_loss = tf.reduce_sum(input_tensor=similarity_loss, axis=1)
            similarity_loss = tf.reduce_sum(input_tensor=similarity_loss, axis=0)
            similarity_loss = self.similarity_regularization * similarity_loss
            loss = loss + similarity_loss
        if self.entropy_regularization > 0.0:
            probs = tf.nn.softmax(logits=x, axis=-1)
            logprobs = tf.log(x=tf.maximum(x=probs, y=1e-6))
            entropy_loss = -tf.reduce_sum(input_tensor=(probs * logprobs), axis=1)
            entropy_loss = tf.reduce_sum(input_tensor=entropy_loss, axis=0)
            entropy_loss *= -self.entropy_regularization
            loss = loss + entropy_loss
        return loss
