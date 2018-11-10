import numpy as np
import pandas as pd
import tensorflow as tf

from .classifier import Classifier
from ..module import Module
from ..optimizers import Optimizer
from ..transformations import transformation_modules
from ..values import CategoricalValue, identify_value


class BasicClassifier(Classifier):

    def __init__(
        self, data, target_label=None,
        # architecture
        network='resnet',
        # hyperparameters
        capacity=64, depth=2, learning_rate=3e-4, weight_decay=1e-5, batch_size=64
    ):
        super().__init__(name='classifier')

        if target_label is None:
            self.target_label = data.columns[-1]
        else:
            self.target_label = target_label

        self.network_type = network
        self.capacity = capacity
        self.depth = depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

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
            module=self.network_type, modules=transformation_modules, name='encoder',
            input_size=input_size, layer_sizes=[self.capacity for _ in range(self.depth)],
            weight_decay=1e-5
        )

        self.output = self.add_module(
            module='dense', modules=transformation_modules, name='output',
            input_size=self.encoder.size(), output_size=output_size, batchnorm=False,
            activation='none'
        )

        # https://twitter.com/karpathy/status/801621764144971776  ;-)
        self.optimizer = self.add_module(
            module=Optimizer, name='optimizer', algorithm='adam', learning_rate=self.learning_rate,
            clip_gradients=1.0
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            network=self.network_type, capacity=self.capacity, depth=self.depth,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            batch_size=self.batch_size
        )
        return spec

    def get_value(self, name, dtype, data):
        return identify_value(module=self, name=name, dtype=dtype, data=data)

    def tf_train_iteration(self, feed=None):
        if feed is None:
            feed = dict()
        xs = list()
        for value in self.input_values:
            for label in value.trainable_labels():
                x = value.input_tensor(feed=feed.get(label))
                xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.output.transform(x=x)
        loss = self.target_value.loss(x=x, feed=feed.get(self.target_value.name))
        losses = tf.losses.get_regularization_losses(scope=None)
        losses.append(loss)
        loss = tf.add_n(inputs=losses, name=None)
        optimized = self.optimizer.optimize(loss=loss)
        return loss, optimized

    def tf_initialize(self):
        super().tf_initialize()

        # learn
        self.loss, self.optimized = self.train_iteration()

        # learn from file
        num_iterations = tf.placeholder(dtype=tf.int64, shape=(), name='num-iterations')
        assert 'num_iterations' not in Module.placeholders
        Module.placeholders['num_iterations'] = num_iterations
        filenames = tf.placeholder(dtype=tf.string, shape=(None,), name='filenames')
        assert 'filenames' not in Module.placeholders
        Module.placeholders['filenames'] = filenames
        dataset = tf.data.TFRecordDataset(
            filenames=filenames, compression_type='GZIP', buffer_size=1000000
            # num_parallel_reads=None
        )
        dataset = dataset.shuffle(buffer_size=100000, seed=None, reshuffle_each_iteration=True)
        dataset = dataset.repeat(count=None)
        features = {  # critically assumes max one trainable label
            label: value.feature() for value in self.values for label in value.trainable_labels()
        }
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=False)
        dataset = dataset.map(
            map_func=(lambda serialized: tf.parse_example(
                serialized=serialized, features=features, name=None, example_names=None
            )), num_parallel_calls=None
        )
        dataset = dataset.prefetch(buffer_size=1)
        self.iterator = dataset.make_initializable_iterator(shared_name=None)

        def cond(iteration, loss):
            return iteration < num_iterations

        def body(iteration, loss):
            loss, optimized = self.train_iteration(feed=self.iterator.get_next())
            with tf.control_dependencies(control_inputs=(optimized,)):
                iteration += 1
            return iteration, loss

        loss, optimized = self.train_iteration(feed=self.iterator.get_next())
        with tf.control_dependencies(control_inputs=(optimized,)):
            iteration = tf.constant(value=1, dtype=tf.int64, shape=(), verify_shape=False)
            self.optimized_fromfile, self.loss_fromfile = tf.while_loop(
                cond=cond, body=body, loop_vars=(iteration, loss)
            )

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

    def learn(self, num_iterations, data=None, filenames=None, verbose=0):
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
            for iteration in range(num_iterations):
                batch = np.random.randint(num_data, size=self.batch_size)
                feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                loss, _ = self.run(fetches=fetches, feed_dict=feed_dict)
                if verbose > 0 and iteration % verbose + 1 == verbose:
                    print('{iteration}: {loss:1.2e}'.format(iteration=(iteration + 1), loss=loss))

        else:
            fetches = self.iterator.initializer
            feed_dict = {'filenames': filenames}
            self.run(fetches=fetches, feed_dict=feed_dict)
            fetches = (self.loss_fromfile, self.optimized_fromfile)
            if verbose == 0:
                feed_dict = dict(num_iterations=num_iterations, feed_dict=feed_dict)
            else:
                assert num_iterations % verbose == 0
                for iteration in range(num_iterations // verbose):
                    feed_dict = dict(num_iterations=verbose)
                    loss, _ = self.run(fetches=fetches, feed_dict=feed_dict)
                    print('{iteration}: {loss:1.2e}'.format(
                        iteration=((iteration + 1) * verbose), loss=loss
                    ))

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
