"""This module implements BasicSynthesizer."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ks_2samp

from .module import Module, tensorflow_name_scoped
from synthesized.common.synthesizer import Synthesizer
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

        vae_values = list()
        for name, dtype in zip(data.dtypes.axes[0], data.dtypes):
            value = self.get_value(name=name, dtype=dtype, data=data)
            if value is not None:
                value.extract(data=data)
                self.values.append(value)
                if name != self.identifier_label:
                    self.value_output_sizes.append(value.output_size())
                    input_size += value.input_size()
                    output_size += value.output_size()
                    if value.input_size() > 0:
                        vae_values.append(value)

        self.vae = self.add_module(module='vae', name='vae', values=vae_values)

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
        xs = dict()
        for value in self.values:
            if value.name != self.identifier_label and value.input_size() > 0:
                x = value.input_tensor(feed=feed)
                xs[value.name] = x

        loss, optimized = self.vae.learn(xs=xs)
        with tf.control_dependencies(control_inputs=[optimized]):
            optimized = Module.global_step.assign_add(delta=1, use_locking=False, read_value=False)
        return loss, optimized

    def module_initialize(self):
        super().module_initialize()

        # learn
        self.loss, self.optimized = self.train_iteration()

        # synthesize
        num_synthesize = tf.placeholder(dtype=tf.int64, shape=(), name='num-synthesize')
        assert 'num_synthesize' not in Module.placeholders
        Module.placeholders['num_synthesize'] = num_synthesize
        self.synthesized = self.vae.synthesize(n=num_synthesize)

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
