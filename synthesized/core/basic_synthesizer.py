from collections import OrderedDict
from random import randrange

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ks_2samp

from .encodings import encoding_modules
from .module import Module, tensorflow_name_scoped
from .optimizers import Optimizer
from .synthesizer import Synthesizer
from .transformations import transformation_modules
from .values import identify_value


class BasicSynthesizer(Synthesizer):

    def __init__(
        self, data, exclude_encoding_loss=False, summarizer=False,
        # architecture
        network='resnet', encoding='variational',
        # hyperparameters
        capacity=128, depth=2, lstm_mode=0, learning_rate=3e-4, encoding_beta=1e-3,
        weight_decay=1e-5, batch_size=64,
        # person
        gender_label=None, name_label=None, firstname_label=None, lastname_label=None,
        email_label=None,
        # address
        postcode_label=None, street_label=None, address_label=None, postcode_regex=None,
        # identifier
        identifier_label=None, condition_labels=()
    ):
        super().__init__(name='synthesizer', summarizer=summarizer)

        self.exclude_encoding_loss = exclude_encoding_loss

        self.network_type = network
        self.encoding_type = encoding
        self.capacity = capacity
        self.depth = depth
        self.lstm_mode = lstm_mode
        self.learning_rate = learning_rate
        self.encoding_beta = encoding_beta
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # person
        self.person_value = None
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        # address
        self.address_value = None
        self.postcode_label = postcode_label
        self.street_label = street_label
        self.address_label = address_label
        self.postcode_regex = postcode_regex
        # identifier
        self.identifier_value = None
        self.identifier_label = identifier_label
        self.condition_labels = tuple(condition_labels)
        # date
        self.date_value = None

        # history
        self.loss_history = list()
        self.ks_distance_history = list()

        self.values = list()
        self.value_output_sizes = list()
        input_size = 0
        output_size = 0
        condition_size = 0

        for name, dtype in zip(data.dtypes.axes[0], data.dtypes):
            value = self.get_value(name=name, dtype=dtype, data=data)
            if name == self.identifier_label:
                value.extract(data=data)
                self.values.append(value)
                self.value_output_sizes.append(value.output_size())
            elif name in self.condition_labels:
                value.extract(data=data)
                self.values.append(value)
                condition_size += value.input_size()
            elif value is not None:
                value.extract(data=data)
                self.values.append(value)
                self.value_output_sizes.append(value.output_size())
                input_size += value.input_size()
                output_size += value.output_size()

        self.encoder = self.add_module(
            module=self.network_type, modules=transformation_modules, name='encoder',
            input_size=input_size, layer_sizes=[self.capacity for _ in range(self.depth)],
            weight_decay=self.weight_decay
        )

        if self.lstm_mode == 2:
            encoding_type = 'rnn_' + self.encoding_type
        else:
            encoding_type = self.encoding_type
        self.encoding = self.add_module(
            module=encoding_type, modules=encoding_modules, name='encoding',
            input_size=self.encoder.size(), encoding_size=self.capacity,
            condition_size=condition_size, beta=self.encoding_beta
        )

        if self.lstm_mode == 2 or self.identifier_label is None:
            self.modulation = None
        else:
            self.modulation = self.add_module(
                module='modulation', modules=transformation_modules, name='modulation',
                input_size=self.capacity, condition_size=self.identifier_value.embedding_size
            )

        if self.lstm_mode == 1:
            self.lstm = self.add_module(
                module='lstm', modules=transformation_modules, name='lstm',
                input_size=self.encoding.size(), output_size=self.capacity
            )
            input_size = self.lstm.size()
        else:
            self.lstm = None
            input_size = self.encoding.size()

        self.decoder = self.add_module(
            module=self.network_type, modules=transformation_modules, name='decoder',
            input_size=input_size, layer_sizes=[self.capacity for _ in range(self.depth)],
            weight_decay=self.weight_decay
        )

        self.output = self.add_module(
            module='dense', modules=transformation_modules, name='output',
            input_size=self.decoder.size(), output_size=output_size, batchnorm=False,
            activation='none'
        )

        self.optimizer = self.add_module(
            module=Optimizer, name='optimizer', algorithm='adam', learning_rate=self.learning_rate,
            clip_gradients=1.0
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            network=self.network_type, encoding=self.encoding_type, capacity=self.capacity,
            depth=self.depth, lstm_mode=self.lstm_mode, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, batch_size=self.batch_size
        )
        return spec

    def get_value(self, name, dtype, data):
        return identify_value(module=self, name=name, dtype=dtype, data=data)

    def encode(self, data):
        for value in self.values:
            data = value.encode(data=data)
        return data

    def preprocess(self, data):
        for value in self.values:
            data = value.preprocess(data=data)
        return data

    @tensorflow_name_scoped
    def train_iteration(self, feed=None):
        summaries = list()
        xs = list()
        for value in self.values:
            if value.name != self.identifier_label and value.name not in self.condition_labels \
                    and value.input_size() > 0:
                x = value.input_tensor(feed=feed)
                xs.append(x)
        x = tf.concat(values=xs, axis=1)
        x = self.encoder.transform(x=x)
        condition = list()
        for value in self.values:
            if value.name in self.condition_labels:
                condition.append(value.input_tensor(feed=feed))
        x, encoding, encoding_loss = self.encoding.encode(
            x=x, condition=condition, encoding_plus_loss=True
        )
        encoding_mean, encoding_variance = tf.nn.moments(
            x=encoding, axes=(0, 1), shift=None, keep_dims=False
        )
        summaries.append(tf.contrib.summary.scalar(
            name='encoding-mean', tensor=encoding_mean, family=None, step=None
        ))
        summaries.append(tf.contrib.summary.scalar(
            name='encoding-variance', tensor=encoding_variance, family=None, step=None
        ))
        summaries.append(tf.contrib.summary.scalar(
            name='encoding-loss', tensor=encoding_loss, family=None, step=None
        ))
        if self.lstm_mode == 2 and self.identifier_label is not None:
            update = self.identifier_value.input_tensor()
            with tf.control_dependencies(control_inputs=(update,)):
                x = x + 0.0  # trivial operation to enforce dependency
        elif self.lstm_mode == 1:
            if self.identifier_label is None:
                x = self.lstm.transform(x=x)
            else:
                state = self.identifier_value.input_tensor()
                x = self.lstm.transform(x=x, state=state[0])
        elif self.lstm_mode == 0 and self.identifier_label is not None:
            condition = self.identifier_value.input_tensor()
            x = self.modulation.transform(x=x, condition=condition)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        losses = OrderedDict()
        if not self.exclude_encoding_loss:
            losses['encoding'] = encoding_loss
        for value, x in zip(self.values, xs):
            loss = value.loss(x=x, feed=feed)
            if loss is not None:
                losses[value.name] = loss
                summaries.append(tf.contrib.summary.scalar(
                    name=(value.name + '-loss'), tensor=loss, family=None, step=None
                ))
        reg_losses = tf.losses.get_regularization_losses(scope=None)
        if len(reg_losses) > 0:
            regularization_loss = tf.add_n(inputs=reg_losses)
            summaries.append(tf.contrib.summary.scalar(
                name='regularization-loss', tensor=regularization_loss, family=None, step=None
            ))
            losses['regularization'] = regularization_loss
        loss = tf.add_n(inputs=list(losses.values()))
        summaries.append(tf.contrib.summary.scalar(
            name='loss', tensor=loss, family=None, step=None
        ))
        optimized, gradient_norms = self.optimizer.optimize(loss=loss, gradient_norms=True)
        for name, gradient_norm in gradient_norms.items():
            summaries.append(tf.contrib.summary.scalar(
                name=(name + '-gradient-norm'), tensor=gradient_norm, family=None, step=None
            ))
        with tf.control_dependencies(control_inputs=([optimized] + summaries)):
            optimized = Module.global_step.assign_add(delta=1, use_locking=False, read_value=False)
        return losses, loss, optimized

    def module_initialize(self):
        super().module_initialize()

        # learn
        self.losses, self.loss, self.optimized = self.train_iteration()

        # learn from file
        num_iterations = tf.placeholder(dtype=tf.int64, shape=(), name='num-iterations')
        assert 'num_iterations' not in Module.placeholders
        Module.placeholders['num_iterations'] = num_iterations
        filenames = tf.placeholder(dtype=tf.string, shape=(None,), name='filenames')
        assert 'filenames' not in Module.placeholders
        Module.placeholders['filenames'] = filenames
        dataset = tf.data.TFRecordDataset(
            filenames=filenames, compression_type='GZIP', buffer_size=100000
            # num_parallel_reads=None
        )
        # dataset = dataset.cache(filename='')
        # filename: A tf.string scalar tf.Tensor, representing the name of a directory on the filesystem to use for caching tensors in this Dataset. If a filename is not provided, the dataset will be cached in memory.
        dataset = dataset.shuffle(buffer_size=100000, seed=None, reshuffle_each_iteration=True)
        dataset = dataset.repeat(count=None)
        features = dict()
        for value in self.values:
            features.update(value.features())
        # better performance after batch
        # dataset = dataset.map(
        #     map_func=(lambda serialized: tf.parse_single_example(
        #         serialized=serialized, features=features, name=None, example_names=None
        #     )), num_parallel_calls=None
        # )
        dataset = dataset.batch(batch_size=self.batch_size)  # drop_remainder=False
        dataset = dataset.map(
            map_func=(lambda serialized: tf.parse_example(
                serialized=serialized, features=features, name=None, example_names=None
            )), num_parallel_calls=None
        )
        dataset = dataset.prefetch(buffer_size=1)
        self.iterator = dataset.make_initializable_iterator(shared_name=None)

        def cond(iteration, losses, loss):
            return iteration < num_iterations

        def body(iteration, losses, loss):
            losses, loss, optimized = self.train_iteration(feed=self.iterator.get_next())
            with tf.control_dependencies(control_inputs=(optimized, loss)):
                iteration += 1
            return iteration, losses, loss

        losses, loss, optimized = self.train_iteration(feed=self.iterator.get_next())
        with tf.control_dependencies(control_inputs=(optimized,)):
            iteration = tf.constant(value=1, dtype=tf.int64, shape=(), verify_shape=False)
            self.optimized_fromfile, self.losses_fromfile, self.loss_fromfile = tf.while_loop(
                cond=cond, body=body, loop_vars=(iteration, losses, loss)
            )

        # synthesize
        num_synthesize = tf.placeholder(dtype=tf.int64, shape=(), name='num-synthesize')
        assert 'num_synthesize' not in Module.placeholders
        Module.placeholders['num_synthesize'] = num_synthesize
        condition = list()
        for value in self.values:
            if value.name in self.condition_labels:
                condition.append(value.input_tensor())
        x = self.encoding.sample(n=num_synthesize, condition=condition)
        if self.lstm_mode == 2 and self.identifier_label is not None:
            identifier = self.identifier_value.next_identifier()
            identifier = tf.tile(input=identifier, multiples=(num_synthesize,))
        elif self.lstm_mode == 1:
            if self.identifier_label is None:
                x = self.lstm.transform(x=x)
            else:
                identifier, state = self.identifier_value.next_identifier_embedding()
                identifier = tf.tile(input=identifier, multiples=(num_synthesize,))
                x = self.lstm.transform(x=x, state=state)
        elif self.lstm_mode == 0 and self.identifier_label is not None:
            identifier, condition = self.identifier_value.random_value(n=num_synthesize)
            x = self.modulation.transform(x=x, condition=condition)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None)
        self.synthesized = dict()
        if self.identifier_value is not None:
            self.synthesized[self.identifier_label] = identifier
        for value, x in zip(self.values, xs):
            xs = value.output_tensors(x=x)
            for label, x in xs.items():
                self.synthesized[label] = x

    def learn(self, num_iterations=2500, data=None, filenames=None, verbose=0):
        if self.lstm_mode != 0 and (
            self.identifier_label is not None or len(self.condition_labels) > 0
        ):
            raise NotImplementedError

        if (data is None) is (filenames is None):
            raise NotImplementedError

        if filenames is None:
            data = self.preprocess(data=data.copy())

            num_data = len(data)
            data = {
                label: data[label].get_values() for value in self.values
                for label in value.input_labels()
            }

            fetches = self.optimized
            if verbose > 0:
                verbose_fetches = dict(self.losses)
                verbose_fetches['loss'] = self.loss

            for iteration in range(num_iterations):
                if self.lstm_mode != 0:
                    start = randrange(max(num_data - self.batch_size, 1))
                    batch = np.arange(start, max(start + self.batch_size, num_data))
                else:
                    batch = np.random.randint(num_data, size=self.batch_size)

                feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                self.run(fetches=fetches, feed_dict=feed_dict)

                if verbose > 0 and (
                    iteration == 0 or iteration + 1 == verbose // 2 or
                    iteration % verbose + 1 == verbose
                ):
                    if self.lstm_mode != 0:
                        start = randrange(max(num_data - 1024, 1))
                        batch = np.arange(start, max(start + 1024, num_data))
                    else:
                        batch = np.random.randint(num_data, size=1024)

                    feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                    fetched = self.run(fetches=verbose_fetches, feed_dict=feed_dict)
                    self.log_metrics(data, fetched, iteration)

        else:
            if verbose > 0:
                raise NotImplementedError
            fetches = self.iterator.initializer
            feed_dict = dict(filenames=filenames)
            self.run(fetches=fetches, feed_dict=feed_dict)
            fetches = self.optimized_fromfile
            feed_dict = dict(num_iterations=num_iterations)
            self.run(fetches=fetches, feed_dict=feed_dict)
            # assert num_iterations % verbose == 0
            # for iteration in range(num_iterations // verbose):
            #     feed_dict = dict(num_iterations=verbose)
            #     fetched = self.run(fetches=fetches, feed_dict=feed_dict)
            #     self.log_metrics(data, fetched, iteration)

    def log_metrics(self, data, fetched, iteration):
        print('\niteration: {}'.format(iteration + 1))
        print('loss: total={loss:1.2e} ({losses})'.format(
            iteration=(iteration + 1), loss=fetched['loss'], losses=', '.join(
                '{name}={loss}'.format(name=name, loss=fetched[name])
                for name in self.losses
            )
        ))
        self.loss_history.append({name: fetched[name] for name in self.losses})

        synthesized = self.synthesize(10000)
        synthesized = self.preprocess(data=synthesized)
        dist_by_col = [(col, ks_2samp(data[col], synthesized[col].get_values())[0]) for col in data.keys()]
        avg_dist = np.mean([dist for (col, dist) in dist_by_col])
        dists = ', '.join(['{col}={dist:.2f}'.format(col=col, dist=dist) for (col, dist) in dist_by_col])
        print('KS distances: avg={avg_dist:.2f} ({dists})'.format(avg_dist=avg_dist, dists=dists))
        self.ks_distance_history.append(dict(dist_by_col))

    def get_loss_history(self):
        return pd.DataFrame.from_records(self.loss_history)

    def get_ks_distance_history(self):
        return pd.DataFrame.from_records(self.ks_distance_history)

    def synthesize(self, n, condition=None):
        if self.lstm_mode != 0 and (
            self.identifier_label is not None or len(self.condition_labels) > 0
        ):
            raise NotImplementedError

        fetches = self.synthesized

        feed_dict = {'num_synthesize': n % 1024}
        if condition is not None:
            feed_dict.update(condition)
        synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
        columns = [
            label for value in self.values if value.name not in self.condition_labels
            for label in value.output_labels()
        ]
        synthesized = pd.DataFrame.from_dict(synthesized)[columns]

        feed_dict['num_synthesize'] = 1024
        for _ in range(n // 1024):
            other = self.run(fetches=fetches, feed_dict=feed_dict)
            other = pd.DataFrame.from_dict(other)[columns]
            synthesized = synthesized.append(other, ignore_index=True)

        for value in self.values:
            if value.name not in self.condition_labels:
                synthesized = value.postprocess(data=synthesized)

        return synthesized
