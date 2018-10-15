import numpy as np
import pandas as pd
import tensorflow as tf

from .classifier import Classifier
from ..module import Module
from ..optimizers import Optimizer
from ..transformations import DenseTransformation, transformation_modules
from ..values import CategoricalValue, identify_value


class BasicClassifier(Classifier):

    def __init__(
        self, data, target_label=None, layers=(64, 64), embedding_size=32, batch_size=64,
        iterations=50000
    ):
        super().__init__(name='classifier')

        if target_label is None:
            self.target_label = data.columns[-1]
        else:
            self.target_label = target_label

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.iterations = iterations

        self.values = list()
        self.input_values = list()
        input_size = 0
        for name, dtype in zip(data.dtypes.axes[0], data.dtypes):
            value = self.get_value(name=name, dtype=dtype, data=data)
            if value is not None:
                value.extract(data=data)
                self.values.append(value)
                if name == self.target_label:
                    if not isinstance(value, CategoricalValue):
                        raise NotImplementedError
                    self.target_value = value
                    output_size = value.output_size()
                elif value is not None:
                    self.input_values.append(value)
                    input_size += value.input_size()

        self.encoder = self.add_module(
            module='mlp', modules=transformation_modules, name='encoder',
            input_size=input_size, layer_sizes=layers
        )

        self.output = self.add_module(
            module=DenseTransformation, name='output', input_size=self.encoder.size(),
            output_size=output_size, batchnorm=False, activation='none'
        )

        # https://twitter.com/karpathy/status/801621764144971776  ;-)
        self.optimizer = self.add_module(
            module=Optimizer, name='optimizer', algorithm='adam', learning_rate=3e-4,
            clip_gradients=1.0
        )

    def specification(self):
        spec = super().specification()
        # TODO: values?
        spec.update(
            encoder=self.encoder, batch_size=self.batch_size, iterations=self.iterations
        )
        return spec

    def get_value(self, name, dtype, data):
        return identify_value(module=self, name=name, dtype=dtype, data=data)

    def tf_initialize(self):
        super().tf_initialize()

        # learn
        xs = list()
        for value in self.input_values:
            for label in value.trainable_labels():
                x = value.input_tensor()
                xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.output.transform(x=x)
        loss = self.target_value.loss(x=x)
        losses = tf.losses.get_regularization_losses(scope=None)
        losses.append(loss)
        self.loss = tf.add_n(inputs=losses, name=None)
        self.optimized = self.optimizer.optimize(loss=self.loss)

        # learn from file
        filenames = tf.placeholder(dtype=tf.string, shape=(None,), name='filenames')
        assert 'filenames' not in Module.placeholders
        Module.placeholders['filenames'] = filenames
        dataset = tf.data.TFRecordDataset(
            filenames=filenames, compression_type='GZIP', buffer_size=1000000
            # num_parallel_reads=None
        )
        # dataset = dataset.cache(filename='')
        # filename: A tf.string scalar tf.Tensor, representing the name of a directory on the filesystem to use for caching tensors in this Dataset. If a filename is not provided, the dataset will be cached in memory.
        dataset = dataset.shuffle(buffer_size=100000, seed=None, reshuffle_each_iteration=True)
        dataset = dataset.repeat(count=None)
        features = {  # critically assumes max one trainable label
            label: value.feature() for value in self.values for label in value.trainable_labels()
        }
        # better performance after batch
        # dataset = dataset.map(
        #     map_func=(lambda serialized: tf.parse_single_example(
        #         serialized=serialized, features=features, name=None, example_names=None
        #     )), num_parallel_calls=None
        # )
        dataset = dataset.batch(batch_size=self.batch_size)  # newer: drop_remainder=False
        dataset = dataset.map(
            map_func=(lambda serialized: tf.parse_example(
                serialized=serialized, features=features, name=None, example_names=None
            )), num_parallel_calls=None
        )
        dataset = dataset.prefetch(buffer_size=1)
        self.iterator = dataset.make_initializable_iterator(shared_name=None)
        next_values = self.iterator.get_next()
        xs = list()
        for value in self.input_values:
            for label in value.trainable_labels():
                x = value.input_tensor(feed=next_values[label])
                xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.output.transform(x=x)
        loss = self.target_value.loss(x=x, feed=next_values[self.target_value.name])
        losses = tf.losses.get_regularization_losses(scope=None)
        losses.append(loss)
        self.loss_fromfile = tf.add_n(inputs=losses, name=None)
        self.optimized_fromfile = self.optimizer.optimize(loss=self.loss_fromfile)

        # classify
        xs = list()
        for value in self.input_values:
            for label in value.trainable_labels():
                x = value.input_tensor()
                xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = self.target_value.output_tensors(x=x)
        if len(xs) != 1:
            raise NotImplementedError
        self.classified = dict()
        for label, x in xs.items():
            self.classified[label] = x

    def learn(self, data=None, filenames=None, verbose=0):
        if (data is None) is (filenames is None):
            raise NotImplementedError

        # TODO: increment global step
        if filenames is None:
            for value in self.values:
                data = value.preprocess(data=data)
            num_data = len(data)
            data = {
                label: data[label].get_values() for value in self.values
                for label in value.trainable_labels()
            }
            fetches = (self.loss, self.optimized)
            for i in range(self.iterations):
                batch = np.random.randint(num_data, size=self.batch_size)
                feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                loss, _ = self.run(fetches=fetches, feed_dict=feed_dict)
                if verbose and i % verbose + 1 == verbose:
                    print('{iteration}: {loss:1.2e}'.format(iteration=(i + 1), loss=loss))

        else:
            fetches = self.iterator.initializer
            feed_dict = {'filenames': filenames}
            self.run(fetches=fetches, feed_dict=feed_dict)
            fetches = (self.loss_fromfile, self.optimized_fromfile)
            # TODO: while loop for training
            for i in range(self.iterations):
                loss, _ = self.run(fetches=fetches)
                if verbose and i % verbose + 1 == verbose:
                    print('{iteration}: {loss:1.2e}'.format(iteration=(i + 1), loss=loss))

    def classify(self, data):
        for value in self.input_values:
            data = value.preprocess(data=data)
        num_data = len(data)  # not needed
        fetches = self.classified
        data = {
            label: data[label].get_values() for value in self.input_values
            for label in value.trainable_labels()
        }
        classified = list()
        for i in range(num_data // self.batch_size):
            feed_dict = {
                label: value_data[i * self.batch_size: (i + 1) * self.batch_size]
                for label, value_data in data.items()
            }
            classified.append(self.run(fetches=fetches, feed_dict=feed_dict))
        i = num_data // self.batch_size
        if i * self.batch_size < num_data:
            feed_dict = {
                label: value_data[i * self.batch_size:] for label, value_data in data.items()
            }
            classified.append(self.run(fetches=fetches, feed_dict=feed_dict))
        classified = {
            label: np.concatenate([c[label] for c in classified]) for label in classified[0]
        }
        classified = pd.DataFrame.from_dict(classified)
        classified = self.target_value.postprocess(data=classified)
        return classified
