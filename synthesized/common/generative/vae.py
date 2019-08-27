from collections import OrderedDict
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .generative import Generative
from ..module import tensorflow_name_scoped
from ..values import Value


class VAE(Generative):
    """Variational auto-encoder.

    The VAE consists of an NN-parametrized input-conditioned encoder distribution q(z|x), a latent
    prior distribution p'(z), optional additional input conditions c, and an NN-parametrized
    latent-conditioned decoder distribution p(y|z,c). The optimized loss consists of the
    reconstruction loss per value, the KL loss, and the regularization loss. The input and output
    are concatenated / split tensors per value. The encoder and decoder network use the same
    hyperparameters.
    """

    def __init__(
        self, name: str, values: List[Value], conditions: List[Value],
        # Latent distribution
        distribution: str, latent_size: int,
        # Encoder and decoder network
        network: str, capacity: int, depth: int, batchnorm: bool, activation: str,
        # Optimizer
        optimizer: str, learning_rate: float, decay_steps: int, decay_rate: float,
        initial_boost: bool, clip_gradients: float,
        # Beta KL loss coefficient
        beta: float,
        # Weight decay
        weight_decay: float,
        summarize: bool = False
    ):
        super().__init__(name=name, values=values, conditions=conditions)

        self.latent_size = latent_size
        self.beta = beta
        self.summarize = summarize

        # Total input and output size of all values
        input_size = 0
        output_size = 0
        for value in self.values:
            input_size += value.learned_input_size()
            output_size += value.learned_output_size()

        # Total condition size
        condition_size = 0
        for value in self.conditions:
            assert value.learned_input_size() > 0
            assert value.learned_input_columns() == value.learned_output_columns()
            condition_size += value.learned_input_size()

        # Encoder: parametrized distribution q(z|x)
        parametrization = dict(
            module=network, layer_sizes=[capacity for _ in range(depth)], batchnorm=batchnorm,
            activation=activation, weight_decay=weight_decay
        )
        self.encoder = self.add_module(
            module='distribution', name='encoder', input_size=input_size, output_size=latent_size,
            distribution=distribution, parametrization=parametrization
        )

        # Decoder: parametrized distribution p(y|z,c)
        parametrization = dict(
            module=network, layer_sizes=[capacity for _ in range(depth)], batchnorm=batchnorm,
            activation=activation, weight_decay=weight_decay
        )
        self.decoder = self.add_module(
            module='distribution', name='decoder',
            input_size=(self.encoder.size() + condition_size), output_size=output_size,
            distribution='deterministic', parametrization=parametrization
        )

        # Optimizer
        self.optimizer = self.add_module(
            module='optimizer', name='optimizer', optimizer=optimizer, parent=self,
            learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, initial_boost=initial_boost,
            clip_gradients=clip_gradients
        )

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            beta=self.beta, encoder=self.encoder.specification(),
            decoder=self.decoder.specification(), optimizer=self.optimizer.specification()
        )
        return spec

    @tensorflow_name_scoped
    def learn(self, xs: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Operation]:
        """Training step for the generative model.

        Args:
            xs: Input tensor per column.

        Returns:
            Dictionary of loss tensors, and optimization operation.

        """
        if len(xs) == 0:
            return dict(), tf.no_op()

        losses: Dict[str, tf.Tensor] = OrderedDict()
        summaries = list()

        # Concatenate input tensors per value
        x = tf.concat(values=[
            value.unify_inputs(xs=[xs[name] for name in value.learned_input_columns()])
            for value in self.values if value.learned_input_size() > 0
        ], axis=1)

        # Encoder q(z|x)
        q = self.encoder.parametrize(x=x)
        if q.reparameterization_type is not tfd.FULLY_REPARAMETERIZED:
            raise NotImplementedError

        # Prior p'(z)
        prior = self.encoder.prior()

        # KL-divergence loss
        kldiv = tfd.kl_divergence(distribution_a=q, distribution_b=prior, allow_nan_stats=False)
        kldiv = tf.reduce_sum(input_tensor=kldiv, axis=1)
        kldiv = tf.reduce_mean(input_tensor=kldiv, axis=0)
        losses['kl-loss'] = self.beta * kldiv
        summaries.append(tf.contrib.summary.scalar(name='kl-divergence', tensor=kldiv))

        # Sample z ~ q(z|x)
        z = q.sample()

        if len(self.conditions) > 0:
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=[xs[name] for name in value.learned_input_columns()])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            z = tf.concat(values=(z, c), axis=1)

        # Decoder p(y|z,c)
        p = self.decoder.parametrize(x=z)

        # Sample y ~ p(y|z,c)
        y = p.sample()

        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )

        # Reconstruction loss per value
        for value, y in zip(self.values, ys):
            losses[value.name + '-loss'] = value.loss(
                y=y, xs=[xs[name] for name in value.learned_output_columns()]
            )

        # Regularization loss
        reg_losses = tf.losses.get_regularization_losses()
        if len(reg_losses) > 0:
            losses['regularization-loss'] = tf.add_n(inputs=reg_losses)

        # Total loss
        total_loss = tf.add_n(inputs=list(losses.values()))
        losses['total-loss'] = total_loss

        # Loss summaries
        for name, loss in losses.items():
            summaries.append(tf.contrib.summary.scalar(name=name, tensor=loss))
            if name != 'total-loss' and name != 'kl-loss':
                summaries.append(tf.contrib.summary.scalar(
                    name=name + '-ratio', tensor=(loss / losses['kl-loss'])
                ))

        if not self.summarize:
            summaries = list()

        # Make sure summary operations are executed
        with tf.control_dependencies(control_inputs=summaries):

            # Optimization step
            optimized = self.optimizer.optimize(
                loss=loss, summarize_gradient_norms=self.summarize
            )

        return losses, optimized

    @tensorflow_name_scoped
    def synthesize(self, n: tf.Tensor, cs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Generate the given number of instances.

        Args:
            n: Number of instances to generate.
            cs: Condition tensor per column.

        Returns:
            Output tensor per column.

        """
        # Prior p'(z)
        prior = self.encoder.prior()

        # Sample z ~ p'(z)
        z = prior.sample(sample_shape=(n,))

        if len(self.conditions) > 0:
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=[cs[name] for name in value.learned_input_columns()])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            z = tf.concat(values=(z, c), axis=1)

        # Decoder p(y|z,c)
        p = self.decoder.parametrize(x=z)

        # Sample y ~ p(y|z,c)
        y = p.sample()

        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )

        # Output tensors per value
        synthesized: Dict[str, tf.Tensor] = OrderedDict()
        for value, y in zip(self.values, ys):
            synthesized.update(zip(value.learned_output_columns(), value.output_tensors(y=y)))

        for value in self.conditions:
            for name in value.learned_output_columns():
                synthesized[name] = cs[name]

        return synthesized
