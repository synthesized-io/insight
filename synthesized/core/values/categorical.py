import tensorflow as tf

from .value import Value
from .. import util
from ..module import Module


class CategoricalValue(Value):

    def __init__(
        self, name, categories, embedding_size, similarity_based=False, temperature=1.0,
        smoothing=0.1, moving_average=True
    ):
        super().__init__(name=name)
        self.categories = sorted(categories)
        self.embedding_size = embedding_size
        self.similarity_based = similarity_based
        self.temperature = temperature
        self.smoothing = smoothing
        self.moving_average = moving_average

    def specification(self):
        spec = super().specification()
        spec.update(
            categories=self.categories, embedding_size=self.embedding_size,
            similarity_based=self.similarity_based, temperature=self.temperature,
            smoothing=self.smoothing, moving_average=self.moving_average
        )
        return spec

    def input_size(self):
        return self.embedding_size

    def output_size(self):
        if self.similarity_based:
            return self.embedding_size
        else:
            return len(self.categories)

    def preprocess(self, data):
        data[self.name] = data[self.name].map(arg=self.categories.index)
        data[self.name] = data[self.name].astype(dtype='int64')
        return data

    def postprocess(self, data):
        data[self.name] = data[self.name].map(arg=self.categories.__getitem__)
        data[self.name] = data[self.name].astype(dtype='category')
        return data

    def feature(self, x=None):
        if x is None:
            return tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=(x,)))

    def tf_initialize(self):
        super().tf_initialize()
        self.placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='input')
        Module.placeholders[self.name] = self.placeholder
        self.embeddings = tf.get_variable(
            name='embeddings', shape=(len(self.categories), self.embedding_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        if self.moving_average:
            self.moving_average = tf.train.ExponentialMovingAverage(
                decay=0.9, num_updates=None, zero_debias=False
            )
        else:
            self.moving_average = None

    def tf_input_tensor(self, feed=None):
        # tensor = tf.one_hot(
        #     indices=self.placeholder, depth=len(self.categories), on_value=1.0, off_value=0.0,
        #     axis=1, dtype=tf.float32
        # )
        x = self.placeholder if feed is None else feed
        x = tf.nn.embedding_lookup(
            params=self.embeddings, ids=x, partition_strategy='mod', validate_indices=True,
            max_norm=None
        )
        return x

    def tf_output_tensor(self, x):
        if self.similarity_based:
            x = tf.expand_dims(input=x, axis=1)
            embeddings = tf.expand_dims(input=self.embeddings, axis=0)
            x = tf.reduce_sum(input_tensor=(x * embeddings), axis=2, keepdims=False)
        x = tf.argmax(input=x, axis=1)
        return x

    def tf_loss(self, x, feed=None):
        target = self.placeholder if feed is None else feed
        if self.moving_average is not None:
            frequency = tf.concat(values=(list(range(len(self.categories))), target), axis=0)
            _, _, frequency = tf.unique_with_counts(x=frequency, out_idx=tf.int32)
            frequency = tf.reshape(tensor=frequency, shape=(len(self.categories),))
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
            indices=target, depth=len(self.categories), on_value=1.0, off_value=0.0, axis=1,
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
        return loss
