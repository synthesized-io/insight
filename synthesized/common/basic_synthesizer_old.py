"""This module implements BasicSynthesizer."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ks_2samp

from .encodings import encoding_modules
from .module import Module, tensorflow_name_scoped
from .optimizers import Optimizer
from synthesized.common.synthesizer import Synthesizer
from .transformations import transformation_modules
from .values import identify_value


class BasicSynthesizer(Synthesizer):
    """The main Synthesizer implementation.

    Synthesizer which can learn to produce basic tabular data with independent rows, that is, no
    temporal or otherwise conditional relation between the rows.
    """

    def __init__(
            self, data, summarizer=False,
            # architecture
            network='resnet', encoding='variational',
            # hyperparameters
            capacity=128, depth=2, learning_rate=3e-4, weight_decay=1e-5, batch_size=64, encoding_beta=0.001,
            # person
            title_label=None, gender_label=None, name_label=None, firstname_label=None, lastname_label=None,
            email_label=None,
            # address
            postcode_label=None, city_label=None, street_label=None,
            address_label=None, postcode_regex=None,
            # identifier
            identifier_label=None
    ):
        """Initialize a new basic synthesizer instance.

        Args:
            data: Data sample which is representative of the target data to generate. Usually, it
                is fine to just use the training data here. Generally, it should exhibit all
                relevant characteristics, so for instance all values a discrete-value column can
                take.
            summarizer: Whether to log TensorBoard summaries (in sub-directory
                "summaries_synthesizer").
            network: Network type: "mlp" or "resnet".
            encoding: Encoding type: "basic", "variational" or "gumbel".
            capacity: Architecture capacity.
            depth: Architecture depth.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            batch_size: Batch size.
            encoding_beta: Encoding loss coefficient.
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

        self.network_type = network
        self.encoding_type = encoding
        self.capacity = capacity
        self.depth = depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

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
        # date
        self.date_value = None

        # history
        self.loss_history = list()
        self.ks_distance_history = list()

        self.values = list()
        self.value_output_sizes = list()
        input_size = 0
        output_size = 0

        for name, dtype in zip(data.dtypes.axes[0], data.dtypes):
            value = self.get_value(name=name, dtype=dtype, data=data)
            if value is not None:
                value.extract(data=data)
                self.values.append(value)
                if name != self.identifier_label:
                    self.value_output_sizes.append(value.output_size())
                    input_size += value.input_size()
                    output_size += value.output_size()

        self.linear_input = self.add_module(
            module='dense', modules=transformation_modules, name='linear-input',
            input_size=input_size, output_size=self.capacity, batchnorm=False, activation='none'
        )

        self.encoder = self.add_module(
            module=self.network_type, modules=transformation_modules, name='encoder',
            input_size=self.linear_input.size(),
            layer_sizes=[self.capacity for _ in range(self.depth)], weight_decay=self.weight_decay
        )

        self.encoding = self.add_module(
            module=self.encoding_type, modules=encoding_modules, name='encoding',
            input_size=self.encoder.size(), encoding_size=self.capacity, beta=encoding_beta
        )

        if self.identifier_value is None:
            self.modulation = None
        else:
            self.modulation = self.add_module(
                module='modulation', modules=transformation_modules, name='modulation',
                input_size=self.capacity, condition_size=self.identifier_value.embedding_size
            )

        self.decoder = self.add_module(
            module=self.network_type, modules=transformation_modules, name='decoder',
            input_size=self.encoding.size(),
            layer_sizes=[self.capacity for _ in range(self.depth)], weight_decay=self.weight_decay
        )

        self.linear_output = self.add_module(
            module='dense', modules=transformation_modules, name='linear-output',
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
            depth=self.depth, learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            batch_size=self.batch_size
        )
        return spec

    def get_value(self, name, dtype, data):
        return identify_value(module=self, name=name, dtype=dtype, data=data)

    def preprocess(self, data):
        for value in self.values:
            data = value.preprocess(data=data)
        return data

    @tensorflow_name_scoped
    def train_iteration(self, feed=None):
        summaries = list()
        xs = list()
        for value in self.values:
            if value.name != self.identifier_label and value.input_size() > 0:
                x = value.input_tensor(feed=feed)
                xs.append(x)
        if len(xs) == 0:
            loss = tf.constant(value=0.0)
            return dict(loss=loss), loss, tf.no_op()
        x = tf.concat(values=xs, axis=1)
        x = self.linear_input.transform(x=x)
        x = self.encoder.transform(x=x)
        x, encoding_loss = self.encoding.encode(x=x, encoding_loss=True)
        summaries.append(tf.contrib.summary.scalar(
            name='encoding-loss', tensor=encoding_loss, family=None, step=None
        ))
        encoding_mean, encoding_variance = tf.nn.moments(
            x=x, axes=(0, 1), shift=None, keep_dims=False
        )
        summaries.append(tf.contrib.summary.scalar(
            name='encoding-mean', tensor=encoding_mean, family=None, step=None
        ))
        summaries.append(tf.contrib.summary.scalar(
            name='encoding-variance', tensor=encoding_variance, family=None, step=None
        ))
        if self.modulation is not None:
            condition = self.identifier_value.input_tensor()
            x = self.modulation.transform(x=x, condition=condition)
        x = self.decoder.transform(x=x)
        x = self.linear_output.transform(x=x)
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
        losses['loss'] = loss
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

        # synthesize
        num_synthesize = tf.placeholder(dtype=tf.int64, shape=(), name='num-synthesize')
        assert 'num_synthesize' not in Module.placeholders
        Module.placeholders['num_synthesize'] = num_synthesize
        x = self.encoding.sample(n=num_synthesize)
        if self.modulation is not None:
            identifier, condition = self.identifier_value.random_value(n=num_synthesize)
            x = self.modulation.transform(x=x, condition=condition)
        x = self.decoder.transform(x=x)
        x = self.linear_output.transform(x=x)
        xs = tf.split(value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None)
        self.synthesized = dict()
        if self.identifier_value is not None:
            self.synthesized[self.identifier_label] = identifier
        for value, x in zip(self.values, xs):
            xs = value.output_tensors(x=x)
            for label, x in xs.items():
                self.synthesized[label] = x

    def learn(
            self, num_iterations: int = 2500, data: pd.DataFrame = None, verbose: int = 0
    ) -> None:
        """Train the generative model on the given data.

        Repeated calls continue training the model, possibly on different data.

        Args:
            num_iterations: The number of training steps (not epochs).
            data: The training data.
            verbose: The frequency, i.e. number of steps, of logging additional information.
        """
        try:
            next(self.learn_async(num_iterations=num_iterations, data=data, verbose=verbose,
                                  yield_every=0))
        except StopIteration:  # since yield_every is 0 we expect an empty generator
            pass

    def learn_async(self, num_iterations=2500, data=None, verbose=0, yield_every=0):
        assert data is not None

        data = self.preprocess(data=data.copy())
        num_data = len(data)
        data = {
            label: data[label].get_values() for value in self.values
            for label in value.input_labels()
        }
        fetches = (self.optimized, self.loss)
        if verbose > 0:
            verbose_fetches = self.losses
        for iteration in range(num_iterations):
            batch = np.random.randint(num_data, size=self.batch_size)
            feed_dict = {label: value_data[batch] for label, value_data in data.items()}
            self.run(fetches=fetches, feed_dict=feed_dict)
            if verbose > 0 and (iteration == 0 or iteration % verbose + 1 == verbose):
                batch = np.random.randint(num_data, size=1024)
                feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                fetched = self.run(fetches=verbose_fetches, feed_dict=feed_dict)
                self.log_metrics(data, fetched, iteration)
            if yield_every > 0 and iteration % yield_every + 1 == yield_every:
                yield iteration

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

    def synthesize(self, n: int) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            n: The number of rows to generate.

        Returns:
            The generated data.

        """
        fetches = self.synthesized
        feed_dict = {'num_synthesize': n % 1024}
        synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
        columns = [label for value in self.values for label in value.output_labels()]
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
            synthesized = value.postprocess(data=synthesized)
        if len(columns) == 0:
            synthesized.pop('_sentinel')
        return synthesized
