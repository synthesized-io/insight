"""This module implements the ScenarioSynthesizer class."""
from collections import OrderedDict
from typing import Callable, List, Dict, Any, Union, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import Synthesizer, Distribution, Functional
from ..common.values import Value
from ..version import deprecated


# TODO: Migrate to TensorFlow2
@deprecated('1.0.0')
class ScenarioSynthesizer(Synthesizer):
    """The scenario synthesizer implementation.

    Synthesizer which can learn from a scenario description to produce basic tabular data with
    independent rows, that is, no temporal or otherwise conditional relation between the rows.
    """

    def __init__(
        self, values: Dict[str, Any], functionals: List[Functional], summarizer_dir: str = None,
        # Prior distribution
        distribution: str = 'normal', latent_size: int = 512,
        # Network
        network: str = 'mlp', capacity: int = 512, depth: int = 2, batch_norm: bool = True,
        activation: str = 'relu',
        # Optimizer
        optimizer: str = 'adam', learning_rate: float = 1e-4, decay_steps: int = 200,
        decay_rate: float = 0.5, initial_boost: bool = True, clip_gradients: float = 1.0,
        # Batch size
        batch_size: int = 1024, synthesis_batch_size: Optional[int] = 1024,
        # Losses
        categorical_weight: float = 1.0, continuous_weight: float = 1.0, weight_decay: float = 0.0
    ):
        """Initialize a new ScenarioSynthesizer instance.

        Args:
            values: A list of value descriptions of the target data to generate.
            functionals: A list of functional descriptions of the target data to generate.
            summarizer: Directory for TensorBoard summaries, automatically creates unique subfolder.
            distribution: Distribution type: "normal".
            latent_size: Latent size.
            network: Network type: "mlp" or "resnet".
            capacity: Architecture capacity.
            depth: Architecture depth.
            batch_norm: Whether to use batch normalization.
            activation: Activation function.
            optimizer: Optimizer.
            learning_rate: Learning rate.
            decay_steps: Learning rate decay steps.
            decay_rate: Learning rate decay rate.
            initial_boost: Learning rate boost for initial steps.
            clip_gradients: Gradient norm clipping.
            categorical_weight: Coefficient for categorical value losses.
            continuous_weight: Coefficient for continuous value losses.
            weight_decay: Weight decay.
        """
        super().__init__(name='synthesizer', summarizer_dir=summarizer_dir)

        categorical_kwargs: Dict[str, Any] = dict()
        continuous_kwargs: Dict[str, Any] = dict()
        nan_kwargs: Dict[str, Any] = dict()
        categorical_kwargs['capacity'] = capacity
        nan_kwargs['capacity'] = capacity
        categorical_kwargs['weight_decay'] = weight_decay
        nan_kwargs['weight_decay'] = weight_decay
        categorical_kwargs['weight'] = categorical_weight
        nan_kwargs['weight'] = categorical_weight
        continuous_kwargs['weight'] = continuous_weight
        categorical_kwargs['temperature'] = 1.0
        categorical_kwargs['moving_average'] = False

        # Values
        self.values: List[Value] = list()
        output_size = 0
        for name, value in values.items():
            if isinstance(value, dict):
                if value['module'] == 'categorical':
                    kwargs = categorical_kwargs
                elif value['module'] == 'continuous':
                    kwargs = continuous_kwargs
                elif value['module'] == 'nan':
                    kwargs = nan_kwargs
            elif value == 'categorical':
                kwargs = categorical_kwargs
            elif value == 'continuous':
                kwargs = continuous_kwargs
            elif value == 'nan':
                kwargs = nan_kwargs
            value = self.add_module(module=value, name=name, **kwargs)
            self.values.append(value)
            output_size += value.learned_output_size()

        # Prior distribution p'(z)
        self.distribution = distribution
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.synthesis_batch_size = synthesis_batch_size

        # Decoder: parametrized distribution p(y|z)
        parametrization = dict(
            module=network, layer_sizes=[capacity for _ in range(depth)], batch_norm=batch_norm,
            activation=activation, weight_decay=weight_decay
        )
        self.decoder = self.add_module(
            module='distribution', name='decoder', input_size=latent_size,
            output_size=output_size, distribution='deterministic', parametrization=parametrization
        )

        # Functionals
        self.functionals: List[tf.Module] = list()
        for functional in functionals:
            functional = self.add_module(module=functional)
            self.functionals.append(functional)

        # Optimizer
        self.optimizer = self.add_module(
            module='optimizer', name='optimizer', optimizer=optimizer,
            learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, initial_boost=initial_boost,
            clip_gradients=clip_gradients
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            values=[value.specification() for value in self.values],
            decoder=self.decoder.specification(),
            functionals=[functional.specification() for functional in self.functionals]
        )
        return spec

    def module_initialize(self):
        super().module_initialize()

        summaries = list()

        # Prior p'(z)
        prior = Distribution.get_prior(distribution=self.distribution, size=self.latent_size)

        # Sample z ~ q(z|x)
        z = prior.sample(sample_shape=(self.num_rows,))

        # Decoder p(y|z)
        p = self.decoder.parametrize(x=z)

        # Sample y ~ p(y|z)
        y = p.sample()

        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )

        # Outputs per value
        self.losses = OrderedDict()
        self.synthesized = OrderedDict()
        for value, y in zip(self.values, ys):

            # Output tensor
            ys = value.output_tensors(y=y)
            self.synthesized.update(zip(value.learned_output_columns(), ys))

            # Distribution loss
            loss = value.distribution_loss(ys=ys)
            self.losses[value.name + '-loss'] = loss

        # Functionals
        for functional in self.functionals:

            # Outputs required for functional
            if functional.required_outputs() == '*':
                samples_args = list(self.synthesized.values())
            else:
                samples_args = [self.synthesized[label] for label in functional.required_outputs()]

            # Functional loss
            self.losses[functional.name + '-loss'] = functional.loss(*samples_args)

        # Regularization loss
        reg_losses = tf.compat.v1.losses.get_regularization_losses()
        if len(reg_losses) > 0:
            self.losses['regularization-loss'] = tf.add_n(inputs=reg_losses)

        # Total loss
        total_loss = tf.add_n(inputs=list(self.losses.values()))
        self.losses['total-loss'] = total_loss

        # Loss summaries
        for name, loss in self.losses.items():
            summaries.append(tf.summary.scalar(name=name, data=loss))

        # Make sure summary operations are executed
        with tf.control_dependencies(control_inputs=summaries):

            # Optimization step
            optimized = self.optimizer.optimize(
                loss=loss, summarize_gradient_norms=(self.summarizer is not None)
            )

        with tf.control_dependencies(control_inputs=[optimized]):
            self.optimized = self.global_step.assign_add(delta=1)

    def learn(
        self, df_train: pd.DataFrame = None, num_iterations: Optional[int] = None,
        callback: Callable[[Synthesizer, int, dict], bool] = Synthesizer.logging,
        callback_freq: int = 0
    ) -> None:
        """Train the generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            df_train: The training data, not used for ScenarioSynthesizer.
            num_iterations: The number of training iterations (not epochs).
            num_samples: The number of samples for which the loss is computed.
            callback: A callback function, e.g. for logging purposes. Takes the synthesizer
                instance, the iteration number, and a dictionary of values (usually the losses) as
                arguments. Aborts training if the return value is True.
            callback_freq: Callback frequency.

        """
        if num_iterations is None:
            raise NotImplementedError

        fetches = self.optimized
        callback_fetches = (self.optimized, self.losses)
        feed_dict = {self.num_rows: np.array(self.batch_size)}

        for iteration in range(1, num_iterations + 1):
            if callback is not None and callback_freq > 0 and (
                iteration == 1 or iteration == num_iterations or iteration % callback_freq == 0
            ):
                _, fetched = self.run(fetches=callback_fetches, feed_dict=feed_dict)
                if callback(self, iteration, fetched) is True:
                    return
            else:
                self.run(fetches=fetches, feed_dict=feed_dict)

    def synthesize(
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:

        columns = [label for value in self.values for label in value.learned_output_columns()]
        if len(columns) == 0:
            return pd.DataFrame([[], ]*num_rows)

        fetches = self.synthesized

        if self.synthesis_batch_size:
            feed_dict = {self.num_rows: (num_rows % self.synthesis_batch_size)}
            synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
            df_synthesized = pd.DataFrame.from_dict(synthesized)[columns]

            feed_dict = {self.num_rows: self.synthesis_batch_size}
            for k in range(num_rows // self.synthesis_batch_size):
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                df_synthesized = df_synthesized.append(
                    pd.DataFrame.from_dict(other)[columns], ignore_index=True
                )
        else:
            feed_dict = {self.num_rows: num_rows}
            synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
            df_synthesized = pd.DataFrame.from_dict(synthesized)[columns]

        for value in self.values:
            df_synthesized = value.postprocess(df=df_synthesized)

        if len(columns) == 0:
            df_synthesized.pop('_sentinel')

        return df_synthesized
