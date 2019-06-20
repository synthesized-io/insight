from collections import OrderedDict
from typing import Callable

import pandas as pd
import tensorflow as tf

from .functionals import functional_modules
from .optimizers import Optimizer
from ..common import identify_value, Module, tensorflow_name_scoped
from ..synthesizer import Synthesizer


class ScenarioSynthesizer(Synthesizer):

    def __init__(
            self, values, functionals, summarizer=False,
            # architecture
            network='resnet',
            # hyperparameters
            capacity=64, depth=4, learning_rate=3e-4, weight_decay=1e-5
    ):
        super().__init__(name='scenario-synthesizer', summarizer=summarizer)

        self.network_type = network
        self.capacity = capacity
        self.depth = depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # history
        self.loss_history = list()

        self.values = list()
        self.value_output_sizes = list()
        output_size = 0
        for name, value in values.items():
            value = self.add_module(module=value, modules=value_modules, name=name)
            self.values.append(value)
            self.value_output_sizes.append(value.output_size())
            output_size += value.output_size()

        self.decoder = self.add_module(
            module=self.network_type, name='decoder',
            input_size=self.capacity, layer_sizes=[self.capacity for _ in range(self.depth)],
            weight_decay=self.weight_decay
        )

        self.output = self.add_module(
            module='dense', name='output',
            input_size=self.decoder.size(), output_size=output_size, batchnorm=False,
            activation='none'
        )

        self.functionals = list()
        for functional in functionals:
            functional = self.add_module(module=functional, modules=functional_modules)
            self.functionals.append(functional)

        self.optimizer = self.add_module(
            module=Optimizer, name='optimizer', algorithm='adam', learning_rate=self.learning_rate,
            clip_gradients=1.0
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            network=self.network_type, capacity=self.capacity, depth=self.depth,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        return spec

    def customized_transform(self, x):
        return x

    def module_initialize(self):
        super().module_initialize()

        # number of rows to synthesize
        num_synthesize = tf.placeholder(dtype=tf.int64, shape=(), name='num-synthesize')
        assert 'num_synthesize' not in Module.placeholders
        Module.placeholders['num_synthesize'] = num_synthesize

        # learn
        summaries = list()
        x = tf.random_normal(
            shape=(num_synthesize, self.capacity), mean=0.0, stddev=1.0, dtype=tf.float32,
            seed=None
        )
        x = self.customized_transform(x=x)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        self.synthesized = OrderedDict()
        self.losses = OrderedDict()
        for value, x in zip(self.values, xs):
            loss = value.distribution_loss(samples=x)
            assert value.name not in self.losses
            if loss is not None:
                self.losses[value.name] = loss
                summaries.append(tf.contrib.summary.scalar(
                    name=(value.name + '-loss'), tensor=loss, family=None, step=None
                ))
            xs = value.output_tensors(x=x)
            for label, x in xs.items():
                self.synthesized[label] = x
        for functional in self.functionals:
            if functional.required_outputs() == '*':
                samples_args = list(self.synthesized.values())
            else:
                samples_args = [self.synthesized[label] for label in functional.required_outputs()]
            loss = functional.loss(*samples_args)
            assert functional.name not in self.losses
            self.losses[functional.name] = loss
            summaries.append(tf.contrib.summary.scalar(
                name=(functional.name + '-loss'), tensor=loss, family=None, step=None
            ))
        reg_losses = tf.losses.get_regularization_losses(scope=None)
        if len(reg_losses) > 0:
            regularization_loss = tf.add_n(inputs=reg_losses)
            summaries.append(tf.contrib.summary.scalar(
                name='regularization-loss', tensor=regularization_loss, family=None, step=None
            ))
            self.losses['regularization'] = regularization_loss
        self.loss = tf.add_n(inputs=list(self.losses.values()))
        assert 'loss' not in self.losses
        self.losses['loss'] = self.loss
        summaries.append(tf.contrib.summary.scalar(
            name='loss', tensor=loss, family=None, step=None
        ))
        optimized, gradient_norms = self.optimizer.optimize(loss=self.loss, gradient_norms=True)
        for name, gradient_norm in gradient_norms.items():
            summaries.append(tf.contrib.summary.scalar(
                name=(name + '-gradient-norm'), tensor=gradient_norm, family=None, step=None
            ))
        with tf.control_dependencies(control_inputs=([optimized] + summaries)):
            self.optimized = Module.global_step.assign_add(
                delta=1, use_locking=False, read_value=True
            )

    def learn(
        self, num_iterations: int, num_samples=1024,
        callback: Callable[[int, dict], None] = Synthesizer.logging, callback_freq: int = 0
    ) -> None:
        """Train the generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            num_iterations: The number of training iterations (not epochs).
            num_samples: The number of samples for which the loss is computed.
            callback: A callback function, e.g. for logging purposes. Aborts training if the return
                value is True.
            callback_freq: Callback frequency.

        """
        feed_dict = {'num_synthesize': num_samples}
        fetches = self.optimized
        callback_fetches = (self.optimized, self.losses)

        for iteration in range(num_iterations):
            if callback is not None and callback_freq > 0 and (
                iteration == 0 or iteration == num_iterations - 1 or iteration % callback_freq == 0
            ):
                _, fetched = self.run(fetches=callback_fetches, feed_dict=feed_dict)
                if callback(iteration, fetched) is True:
                    return
            else:
                self.run(fetches=fetches, feed_dict=feed_dict)

    # def log_metrics(self, fetched, iteration):
    #     print('\niteration: {}'.format(iteration + 1))
    #     print('loss: total={loss:1.2e} ({losses})'.format(
    #         iteration=(iteration + 1), loss=fetched['loss'], losses=', '.join(
    #             '{name}={loss}'.format(name=name, loss=fetched[name])
    #             for name in self.losses
    #         )
    #     ))
    #     self.loss_history.append({name: fetched[name] for name in self.losses})
    #

    def synthesize(self, num_rows: int) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.

        Returns:
            The generated data.

        """
        fetches = self.synthesized
        feed_dict = {'num_synthesize': num_rows % 1024}
        synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
        synthesized = pd.DataFrame.from_dict(synthesized)
        feed_dict = {'num_synthesize': 1024}
        for k in range(num_rows // 1024):
            other = self.run(fetches=fetches, feed_dict=feed_dict)
            other = pd.DataFrame.from_dict(other)
            synthesized = synthesized.append(other, ignore_index=True)
        for value in self.values:
             synthesized = value.postprocess(data=synthesized)
        return synthesized
