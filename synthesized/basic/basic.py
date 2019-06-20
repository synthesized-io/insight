"""This module implements BasicSynthesizer."""
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import identify_value, Module, tensorflow_name_scoped
from ..synthesizer import Synthesizer


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
            value = identify_value(module=self, name=name, dtype=dtype, data=data)
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

    def module_initialize(self):
        super().module_initialize()

        # learn
        xs = dict()
        for value in self.values:
            if value.name != self.identifier_label and value.input_size() > 0:
                xs[value.name] = value.input_tensor()
        self.losses, optimized = self.vae.learn(xs=xs)
        with tf.control_dependencies(control_inputs=[optimized]):
            self.optimized = Module.global_step.assign_add(delta=1)

        # synthesize
        num_synthesize = tf.placeholder(dtype=tf.int64, shape=(), name='num-synthesize')
        assert 'num_synthesize' not in Module.placeholders
        Module.placeholders['num_synthesize'] = num_synthesize
        self.synthesized = self.vae.synthesize(n=num_synthesize)

    def learn(
        self, num_iterations: int, data: pd.DataFrame, callback: Callable[[int, dict], None] = None,
        callback_freq: int = 100, **kwargs
    ) -> None:
        """Train the generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            num_iterations: The number of training iterations (not epochs).
            data: The training data.
            callback: A callback function, e.g. for logging purposes. Aborts training if the return
                value is True.
            callback_freq: Callback frequency.

        """
        data = data.copy()
        for value in self.values:
            data = value.preprocess(data=data)
        num_data = len(data)
        data = {
            label: data[label].get_values() for value in self.values
            for label in value.input_labels()
        }
        fetches = self.optimized
        callback_fetches = (self.optimized, self.losses)

        for iteration in range(num_iterations):
            batch = np.random.randint(num_data, size=self.batch_size)
            feed_dict = {label: value_data[batch] for label, value_data in data.items()}
            if callback is not None and (
                iteration == 0 or iteration == num_iterations - 1 or iteration % callback_freq == 0
            ):
                _, fetched = self.run(fetches=callback_fetches, feed_dict=feed_dict)
                if callback(iteration, fetched) is True:
                    return
            else:
                self.run(fetches=fetches, feed_dict=feed_dict)

    def synthesize(self, num_rows: int) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.

        Returns:
            The generated data.

        """
        fetches = self.synthesized
        feed_dict = {'num_synthesize': num_rows % 1024}
        columns = [label for value in self.values for label in value.output_labels()]
        if len(columns) == 0:
            synthesized = pd.DataFrame(dict(_sentinel=np.zeros((num_rows,))))
        else:
            synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
            synthesized = pd.DataFrame.from_dict(synthesized)[columns]
            feed_dict = {'num_synthesize': 1024}
            for k in range(num_rows // 1024):
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                other = pd.DataFrame.from_dict(other)[columns]
                synthesized = synthesized.append(other, ignore_index=True)
        for value in self.values:
            synthesized = value.postprocess(data=synthesized)
        if len(columns) == 0:
            synthesized.pop('_sentinel')
        return synthesized
