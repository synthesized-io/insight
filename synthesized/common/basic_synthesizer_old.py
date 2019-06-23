"""This module implements BasicSynthesizer."""

from collections import OrderedDict
from random import randrange

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ks_2samp

from .encodings import encoding_modules
from .module import Module, tensorflow_name_scoped
from .optimizers import Optimizer
from synthesized.synthesizer import Synthesizer
from .values import identify_value


class BasicSynthesizer(Synthesizer):
    """The main Synthesizer implementation.

    Synthesizer which can learn to produce basic tabular data with independent rows, that is, no
    temporal or otherwise conditional relation between the rows.
    """

    def __init__(
        self, data, summarizer=None,
        # encoder/decoder
        network_type='mlp', capacity=512, depth=2, layer_type='dense', batchnorm=True,
        activation='relu', weight_decay=1e-5,
        # encoding
        encoding_type='variational', encoding_size=512, encoding_kwargs=dict(beta=0.0005),
        # optimizer
        optimizer='adam', learning_rate=1e-4, decay_steps=200, decay_rate=0.5,
        clip_gradients=1.0,
        batch_size=128,
        # losses
        categorical_weight=1.0, continuous_weight=1.0,
        # categorical
        smoothing=0.0, moving_average=True, similarity_regularization=0.0,
        entropy_regularization=0.1,
        # person
        title_label=None, gender_label=None, name_label=None, firstname_label=None, lastname_label=None,
        email_label=None,
        # address
        postcode_label=None, city_label=None, street_label=None,
        address_label=None, postcode_regex=None,
        # identifier
        identifier_label=None, condition_labels=()
    ):
        """Initialize a new basic synthesizer instance.

        Args:
            data: Data sample which is representative of the target data to generate. Usually, it
                is fine to just use the training data here. Generally, it should exhibit all
                relevant characteristics, so for instance all values a discrete-value column can
                take.
            summarizer: Directory for TensorBoard summaries, automatically creates unique subfolder.
            network_type: Network type: "mlp" or "resnet".
            capacity: Architecture capacity.
            depth: Architecture depth.
            layer_type: Layer type.
            batchnorm: Whether to use batch normalization.
            activation: Activation function.
            weight_decay: Weight decay.
            encoding_type: Encoding type: "basic", "variational" or "gumbel".
            encoding_size: Encoding size.
            encoding_kwargs: Additional arguments to the encoding (beta: encoding loss coefficient).
            optimizer: Optimizer.
            learning_rate: Learning rate.
            decay_steps: Learning rate decay steps.
            decay_rate: Learning rate decay rate.
            clip_gradients: Gradient norm clipping.
            batch_size: Batch size.
            categorical_weight: Coefficient for categorical value losses.
            continuous_weight: Coefficient for continuous value losses.
            smoothing: Smoothing for categorical value distributions.
            moving_average: Whether to use moving average scaling for categorical values.
            similarity_regularization: Similarity regularization coefficient for categorical values.
            entropy_regularization: Entropy regularization coefficient for categorical values.
            title_label: Person title column.
            gender_label: Person gender column.
            name_label: Person combined first and last name column.
            firstname_label: Person first name column.
            lastname_label: Person last name column.
            email_label: Person e-mail address column.
            postcode_label: Address postcode column.
            city_label: Address city column.
            street_label: Address street column.
            address_label: Address combined column.
            postcode_regex: Address postcode regular expression.
            identifier_label: Identifier column.
        """
        super().__init__(name='synthesizer', summarizer=summarizer)

        self.exclude_encoding_loss = False

        self.capacity = capacity
        self.depth = depth
        self.lstm_mode = lstm_mode
        self.learning_rate = learning_rate
        self.encoding_beta = encoding_beta
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.categorical_weight = categorical_weight
        self.continuous_weight = continuous_weight

        self.smoothing = smoothing
        self.moving_average = moving_average
        self.similarity_regularization = similarity_regularization
        self.entropy_regularization = entropy_regularization

        # person
        self.person_value = None
        self.title_label = title_label
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        # address
        self.address_value = None
        self.postcode_label = postcode_label
        self.city_label = city_label
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

        self.linear_input = self.add_module(
            module='dense', name='linear-input',
            input_size=input_size, output_size=capacity, batchnorm=False, activation='none'
        )

        self.encoder = self.add_module(
            module=network_type, name='encoder',
            input_size=self.linear_input.size(), layer_sizes=[capacity for _ in range(depth)],
            layer_type=layer_type, batchnorm=batchnorm, activation=activation,
            weight_decay=weight_decay  # TODO: depths missing
        )

        if self.lstm_mode == 2:
            encoding_type = 'rnn_' + self.encoding_type
            encoding_size = self.capacity  # * 2
        else:
            encoding_type = self.encoding_type
            encoding_size = self.capacity
        self.encoding = self.add_module(
            module=encoding_type, modules=encoding_modules, name='encoding',
            input_size=self.encoder.size(), encoding_size=encoding_size,
            condition_size=condition_size, **encoding_kwargs
        )

        if self.lstm_mode == 2 or self.identifier_label is None:
            self.modulation = None
        else:
            self.modulation = self.add_module(
                module='modulation', name='modulation',
                input_size=capacity, condition_size=self.identifier_value.embedding_size
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
            module=network_type, name='decoder',
            input_size=self.encoding.size(), layer_sizes=[capacity for _ in range(depth)],
            layer_type=layer_type, batchnorm=batchnorm, activation=activation,
            weight_decay=weight_decay  # TODO: depths missing
        )

        self.linear_output = self.add_module(
            module='dense', name='linear-output',
            input_size=self.decoder.size(), output_size=output_size, batchnorm=False,
            activation='none'
        )

        self.optimizer = self.add_module(
            module=Optimizer, name='optimizer', algorithm=optimizer, learning_rate=learning_rate,
            decay_steps=decay_steps, decay_rate=decay_rate, clip_gradients=clip_gradients
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            encoder=self.encoder.specification(), decoder=self.decoder.specification(),
            encoding=self.encoding.specification(), optimizer=self.optimizer.specification(),
            batch_size=self.batch_size
        )
        return spec

    def get_value(self, name, dtype, data):
        return identify_value(module=self, name=name, dtype=dtype, data=data)

    def preprocess(self, data: pd.DataFrame):
        for value in self.values:
            data = value.preprocess(data=data)
        return data

    @tensorflow_name_scoped
    def train_iteration(self, feed: dict=None):
        summaries = list()
        xs = list()
        for value in self.values:
            if value.name != self.identifier_label and value.name not in self.condition_labels \
                    and value.input_size() > 0:
                x = value.input_tensor(feed=feed)
                xs.append(x)
        if len(xs) == 0:
            loss = tf.constant(value=0.0)
            return dict(loss=loss), loss, tf.no_op()
        x = tf.concat(values=xs, axis=1)
        x = self.linear_input.transform(x=x)
        x = self.encoder.transform(x=x)
        condition = list()
        for value in self.values:
            if value.name in self.condition_labels:
                condition.append(value.input_tensor(feed=feed))
        x, encoding, encoding_loss = self.encoding.encode(
            x=x, condition=condition, encoding_plus_loss=True
        )
        summaries.append(tf.contrib.summary.scalar(
            name='loss-encoding', tensor=encoding_loss, family=None, step=None
        ))
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
        x = self.linear_output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        synthesized = OrderedDict()
        # if self.identifier_value is not None:
        #     synthesized[self.identifier_label] = identifier
        for value, x in zip(self.values, xs):
            x = value.output_tensors(x=x)
            for label, x in x.items():
                synthesized[label] = x
        losses = OrderedDict()
        losses['encoding'] = encoding_loss
        for value, x in zip(self.values, xs):
            loss = value.loss(x=x, feed=feed)
            if loss is not None:
                losses[value.name] = loss
                summaries.append(tf.contrib.summary.scalar(
                    name=('loss-' + value.name), tensor=loss, family=None, step=None
                ))
        reg_losses = tf.losses.get_regularization_losses(scope=None)
        if len(reg_losses) > 0:
            regularization_loss = tf.add_n(inputs=reg_losses)
            summaries.append(tf.contrib.summary.scalar(
                name=('loss-regularization'), tensor=regularization_loss, family=None, step=None
            ))
            losses['regularization'] = regularization_loss
        loss = tf.add_n(inputs=list(losses.values()))
        summaries.append(tf.contrib.summary.scalar(
            name='loss', tensor=loss, family=None, step=None
        ))
        optimized, gradient_norms = self.optimizer.optimize(loss=loss, gradient_norms=True)
        summaries.append(tf.contrib.summary.scalar(
            name=('gradient-norm'), tensor=gradient_norms['all']
        ))
        # for name, gradient_norm in gradient_norms.items():
        #     summaries.append(tf.contrib.summary.scalar(
        #         name=(name + '-gradient-norm'), tensor=gradient_norm, family=None, step=None
        #     ))
        for variable in tf.trainable_variables():
            assert variable.name.endswith(':0')
            mean, variance = tf.nn.moments(x=variable, axes=tuple(range(len(variable.shape))))
            summaries.append(tf.contrib.summary.scalar(
                name=(variable.name[:-2] + '-mean'), tensor=mean
            ))
            summaries.append(tf.contrib.summary.scalar(
                name=(variable.name[:-2] + '-variance'), tensor=variance
            ))
        with tf.control_dependencies(control_inputs=([optimized] + summaries)):
            optimized = Module.global_step.assign_add(delta=1, use_locking=False, read_value=False)
        return losses, loss, optimized, synthesized

    def module_initialize(self):
        super().module_initialize()

        # learn
        self.losses, self.loss, self.optimized, self.synthesized_train = self.train_iteration()

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
        x = self.linear_output.transform(x=x)
        xs = tf.split(value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None)
        self.synthesized = OrderedDict()
        if self.identifier_value is not None:
            self.synthesized[self.identifier_label] = identifier
        for value, x in zip(self.values, xs):
            xs = value.output_tensors(x=x)
            for label, x in xs.items():
                self.synthesized[label] = x

    def learn(self, num_iterations=2500, data=None, filenames=None, verbose=0, print_data=False):
        if self.lstm_mode != 0 and (
            self.identifier_label is not None or len(self.condition_labels) > 0
        ):
            raise NotImplementedError

        try:
            next(self.learn_async(num_iterations=num_iterations, data=data, filenames=filenames, verbose=verbose, print_data=print_data, yield_every=0))
        except StopIteration:  # since yield_every is 0 we expect an empty generator
            pass

    def learn_async(
        self, num_iterations=2500, data=None, filenames=None, verbose=0, print_data=0, yield_every=0
    ):
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
                verbose_fetches = dict(losses=self.losses, loss=self.loss)
                if print_data > 0:
                    verbose_fetches['synthesized'] = self.synthesized_train

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
                    self.print_learn_stats(data, batch, fetched, iteration, print_data)
                if yield_every > 0 and iteration % yield_every + 1 == yield_every:
                    yield iteration

        else:
            if verbose > 0:
                raise NotImplementedError
            fetches = self.iterator.initializer
            feed_dict = dict(filenames=filenames)
            self.run(fetches=fetches, feed_dict=feed_dict)
            fetches = (self.optimized_fromfile, self.loss_fromfile)
            feed_dict = dict(num_iterations=num_iterations)
            self.run(fetches=fetches, feed_dict=feed_dict)
            if verbose > 0 and (iteration == 0 or iteration % verbose + 1 == verbose):
                batch = np.random.randint(num_data, size=1024)
                feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                fetched = self.run(fetches=verbose_fetches, feed_dict=feed_dict)
                self.log_metrics(data, fetched, iteration)
            if yield_every > 0 and iteration % yield_every + 1 == yield_every:
                yield iteration

    def print_learn_stats(self, data, batch, fetched, iteration, print_data):
        print('\niteration: {}'.format(iteration + 1))
        print('loss: total={loss:1.2e} ({losses})'.format(
            iteration=(iteration + 1), loss=fetched['loss'], losses=', '.join(
                '{name}={loss}'.format(name=name, loss=loss)
                for name, loss in fetched['losses'].items()
            )
        ))
        self.loss_history.append(fetched['losses'])

        synthesized = self.synthesize(10000)
        synthesized = self.preprocess(data=synthesized)
        dist_by_col = [(col, ks_2samp(data[col], synthesized[col].get_values())[0]) for col in data.keys()]
        avg_dist = np.mean([dist for (col, dist) in dist_by_col])
        dists = ', '.join(['{col}={dist:.2f}'.format(col=col, dist=dist) for (col, dist) in dist_by_col])
        print('KS distances: avg={avg_dist:.2f} ({dists})'.format(avg_dist=avg_dist, dists=dists))
        self.ks_distance_history.append(dict(dist_by_col))

        if print_data > 0:
            print('original data:')
            print(pd.DataFrame.from_dict({key: value[batch] for key, value in data.items()}).head(print_data))
            print('synthesized data:')
            print(pd.DataFrame.from_dict(fetched['synthesized']).head(print_data))

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

        if len(columns) == 0:
            synthesized = pd.DataFrame(dict(_sentinel=np.zeros((n,))))

        else:
            synthesized = pd.DataFrame.from_dict(synthesized)[columns]
            feed_dict = {'num_synthesize': 1024}
            for k in range(n // 1024):
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                other = pd.DataFrame.from_dict(other)[columns]
                synthesized = synthesized.append(other, ignore_index=True)

        for value in self.values:
            if value.name not in self.condition_labels:
                synthesized = value.postprocess(data=synthesized)

        if len(columns) == 0:
            synthesized.pop('_sentinel')

        return synthesized
