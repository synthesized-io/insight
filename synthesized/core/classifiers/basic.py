import numpy as np
import pandas as pd
import tensorflow as tf

from .classifier import Classifier
from ..optimizers import Optimizer
from ..transformations import DenseTransformation, transformation_modules
from ..values import CategoricalValue, identify_value


class BasicClassifier(Classifier):

    def __init__(
        self, data, target_label=None, layers=(64, 64), embedding_size=32, batch_size=64, iterations=50000
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
            x = value.input_tensor()
            if x is not None:
                xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.output.transform(x=x)
        loss = self.target_value.loss(x=x)
        losses = tf.losses.get_regularization_losses(scope=None)
        losses.append(loss)
        self.loss = tf.add_n(inputs=losses, name=None)
        # self.loss = tf.losses.get_total_loss(add_regularization_losses=True, name=None)
        self.optimized = self.optimizer.optimize(loss=self.loss)

        # classify
        xs = list()
        for value in self.input_values:
            x = value.input_tensor()
            if x is not None:
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

    def learn(self, data, verbose=False):
        # TODO: increment global step
        for value in self.values:
            data = value.preprocess(data=data)
        num_data = len(data)
        data = [
            [data[label].get_values() for label in value.trainable_labels()] for value in self.values
        ]
        fetches = (self.loss, self.optimized)
        for i in range(self.iterations):
            batch = np.random.randint(num_data, size=self.batch_size)
            feed_dict = {
                placeholder: d[batch] for value, value_data in zip(self.values, data)
                for placeholder, d in zip(value.placeholders(), value_data)
            }
            loss, _ = self.session.run(
                fetches=fetches, feed_dict=feed_dict, options=None, run_metadata=None
            )
            if verbose and i % verbose + 1 == verbose:
                print('{iteration}: {loss:1.2e}'.format(iteration=(i + 1), loss=loss))

    def classify(self, data):
        for value in self.input_values:
            data = value.preprocess(data=data)
        num_data = len(data)  # not needed
        fetches = self.classified
        # feed_dict = {
        #     placeholder: data[label].get_values() for value in self.input_values
        #     for placeholder, label in zip(value.placeholders(), value.trainable_labels())
        # }
        # classified = self.session.run(
        #     fetches=fetches, feed_dict=feed_dict, options=None, run_metadata=None
        # )
        data = [
            [data[label].get_values() for label in value.trainable_labels()] for value in self.input_values
        ]
        classified = list()
        for i in range(num_data // self.batch_size):
            feed_dict = {
                placeholder: d[i * self.batch_size: (i + 1) * self.batch_size]
                for value, value_data in zip(self.values, data)
                for placeholder, d in zip(value.placeholders(), value_data)
            }
            classified.append(self.session.run(
                fetches=fetches, feed_dict=feed_dict, options=None, run_metadata=None
            ))
        i = num_data // self.batch_size
        if i * self.batch_size < num_data:
            feed_dict = {
                placeholder: d[i * self.batch_size:]
                for value, value_data in zip(self.values, data)
                for placeholder, d in zip(value.placeholders(), value_data)
            }
            classified.append(self.session.run(
                fetches=fetches, feed_dict=feed_dict, options=None, run_metadata=None
            ))
        classified = {label: np.concatenate([c[label] for c in classified]) for label in classified[0]}
        classified = pd.DataFrame.from_dict(classified)
        classified = self.target_value.postprocess(data=classified)
        return classified
