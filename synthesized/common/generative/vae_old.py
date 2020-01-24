from collections import OrderedDict
from itertools import combinations
from typing import Dict, List, Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

from .generative import Generative
from ..module import tensorflow_name_scoped
from ..values import Value, CategoricalValue
from ..transformations import ResnetTransformation, DenseTransformation
from ..encodings import VariationalEncoding
from ..optimizers import Optimizer


class VAEOld(Generative):
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
        distribution: str, latent_size: int, global_step: tf.Variable,
        # Encoder and decoder network
        network: str, capacity: int, num_layers: int, residual_depths: Union[None, int, List[int]], batchnorm: bool,
        activation: str,
        # Optimizer
        optimizer: str, learning_rate: tf.Tensor, decay_steps: int, decay_rate: float,
        initial_boost: int, clip_gradients: float,
        # Beta KL loss coefficient
        beta: float,
        # Weight decay
        weight_decay: float,
        summarize: bool = False, summarize_gradient_norms: bool = False
    ):
        super().__init__(name=name, values=values, conditions=conditions)
        self.global_step = global_step
        self.latent_size = latent_size
        self.beta = beta
        self.summarize = summarize
        self.summarize_gradient_norms = summarize_gradient_norms

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

        self.linear_input = DenseTransformation(
            name='linear-input',
            input_size=input_size, output_size=capacity, batchnorm=False, activation='none'
        )

        if network == 'resnet':
            self.encoder = ResnetTransformation(
                name='encoder', input_size=self.linear_input.size(), depths=residual_depths,
                layer_sizes=[capacity for _ in range(num_layers)], weight_decay=weight_decay
            )
        else:
            # TODO: create a network->class dictionary
            self.encoder = self.add_module(
                module=network, name='encoder',
                input_size=self.linear_input.size(),
                layer_sizes=[capacity for _ in range(num_layers)], weight_decay=weight_decay
            )

        self.encoding = VariationalEncoding(
            name='encoding',
            input_size=self.encoder.size(), encoding_size=self.latent_size, beta=beta
        )

        self.modulation = None

        if network == 'resnet':
            self.decoder = ResnetTransformation(
                name='decoder',
                input_size=(self.encoding.size() + condition_size), depths=residual_depths,
                layer_sizes=[capacity for _ in range(num_layers)], weight_decay=weight_decay
            )
        else:
            self.decoder = self.add_module(
                module=network, name='decoder',
                input_size=(self.encoding.size() + condition_size),
                layer_sizes=[capacity for _ in range(num_layers)], weight_decay=weight_decay
            )

        self.linear_output = DenseTransformation(
            name='linear-output',
            input_size=self.decoder.size(), output_size=output_size, batchnorm=False, activation='none'
        )

        self.optimizer = Optimizer(name='optimizer', optimizer=optimizer, global_step=self.global_step,
            learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
            clip_gradients=clip_gradients, initial_boost=initial_boost
        )

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            beta=self.beta, encoder=self.encoder.specification(),
            decoder=self.decoder.specification(), optimizer=self.optimizer.specification()
        )
        return spec

    def loss(self):
        xs = self.xs

        if len(xs) == 0:
            return dict(), tf.no_op()

        losses: Dict[str, tf.Tensor] = OrderedDict()
        summaries = list()

        with tf.name_scope("Value_Factory"):
            # Concatenate input tensors per value
            x = tf.concat(values=[
                value.unify_inputs(xs=[xs[name] for name in value.learned_input_columns()])
                for value in self.values if value.learned_input_size() > 0
            ], axis=1)

        #################################

        x = self.linear_input(x)
        x = self.encoder(x)
        x = self.encoding(x)

        encoding_loss = tf.identity(self.encoding.losses[0], name='encoding_loss')

        with tf.name_scope("latent_space"):
            tf.summary.histogram(name='mean', data=self.encoding.mean.output, step=self.global_step),
            tf.summary.histogram(name='stddev', data=self.encoding.stddev.output, step=self.global_step),
            tf.summary.histogram(name='posterior_distribution', data=x, step=self.global_step),
            tf.summary.image(
                name='latent_space_correlation',
                data=tf.abs(tf.reshape(tfp.stats.correlation(x), shape=(1, self.latent_size, self.latent_size, 1))),
                step=self.global_step
            )

        with tf.name_scope("Conditions_Factory"):
            if len(self.conditions) > 0:
                # Condition c
                c = tf.concat(values=[
                    value.unify_inputs(xs=[xs[name] for name in value.learned_input_columns()])
                    for value in self.conditions
                ], axis=1)

                # Concatenate z,c
                x = tf.concat(values=(x, c), axis=1)

        x = self.decoder(x)
        y = self.linear_output(x)

        #################################

        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )

        with tf.name_scope("Value_Losses"):
            # Reconstruction loss per value
            for value, y in zip(self.values, ys):
                losses[value.name + '-loss'] = value.loss(
                    y=y, xs=[xs[name] for name in value.learned_output_columns()]
                )

            # Categorical Contingency Plots in tensorboard
            with tf.name_scope("contingency_plots"):
                for (value_a, y_a), (value_b, y_b) in combinations(
                        [(v, tf.one_hot(v.output_tensors(vy)[0], depth=v.num_categories))
                         for v, vy in zip(self.values, ys) if isinstance(v, CategoricalValue)],
                        r=2
                ):
                    tf.summary.image(
                        name=f"{value_a.name}_{value_b.name}",
                        data=tf.expand_dims(tf.cast(tf.reduce_sum(input_tensor=tf.matmul(
                            tf.expand_dims(y_a, axis=-1),
                            tf.expand_dims(y_b, axis=1)
                        ), axis=0, keepdims=True), dtype=tf.float32), axis=-1), step=self.global_step
                    )

            # Regularization loss
            reg_losses = tf.compat.v1.losses.get_regularization_losses()
            if len(reg_losses) > 0:
                regularization_loss = tf.add_n(inputs=reg_losses, name='regularization_loss')
            else:
                regularization_loss = tf.constant(0, dtype=tf.float32)

            # Reconstruction loss
            reconstruction_loss = tf.add_n(inputs=list(losses.values()), name='reconstruction_loss')

        regularization_loss = tf.identity(regularization_loss, name='regularization_loss')
        reconstruction_loss = tf.identity(reconstruction_loss, name='reconstruction_loss')

        losses['encoding-loss'] = encoding_loss
        losses['regularization-loss'] = regularization_loss
        losses['reconstruction-loss'] = reconstruction_loss

        # Total loss
        total_loss = tf.add_n(
            inputs=[encoding_loss, reconstruction_loss, regularization_loss], name='total_loss'
        )

        losses['total-loss'] = total_loss

        # Loss summaries
        for name, loss in losses.items():
            tf.summary.scalar(name=name, data=loss, step=self.global_step)

        return total_loss

    @tf.function
    @tensorflow_name_scoped
    def learn(self, xs: Dict[str, tf.Tensor]) -> None:
        """Training step for the generative model.

        Args:
            xs: Input tensor per column.

        Returns:
            Dictionary of loss tensors, and optimization operation.

        """
        self.xs = xs

        # Optimization step
        optimized = self.optimizer.optimize(
            loss=self.loss, summarize_gradient_norms=self.summarize_gradient_norms, summarize_lr=self.summarize
        )

        return

    @tensorflow_name_scoped
    @tf.function
    def encode(self, xs: Dict[str, tf.Tensor], cs: Dict[str, tf.Tensor]) -> \
            Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Encoding Step for VAE.

        Args:
            xs: Input tensor per column.
            cs: Condition tensor per column.

        Returns:
            Dictionary of Latent space tensor, means and stddevs, dictionary of output tensors per column

        """
        if len(xs) == 0:
            return tf.no_op(), dict()

        # Concatenate input tensors per value
        x = tf.concat(values=[
            value.unify_inputs(xs=[xs[name] for name in value.learned_input_columns()])
            for value in self.values if value.learned_input_size() > 0
        ], axis=1)

        #################################

        x = self.linear_input.transform(x)
        x = self.encoder.transform(x=x)
        z, encoding_loss, mean, std = self.encoding.encode(x=x)

        latent_space = z

        if len(self.conditions) > 0:
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=[cs[name] for name in value.learned_input_columns()])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            z = tf.concat(values=(z, c), axis=1)

        x = self.decoder.transform(x=z)
        y = self.linear_output.transform(x=x)

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

        return {"sample": latent_space, "mean": mean, "std": std}, synthesized

    @tensorflow_name_scoped
    def synthesize(self, n: tf.Tensor, cs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Generate the given number of instances.

        Args:
            n: Number of instances to generate.
            cs: Condition tensor per column.

        Returns:
            Output tensor per column.

        """
        ys = self._synthesize(n=n, cs=cs, num_cs=tf.constant(len(self.conditions), dtype=tf.int64))

        # Output tensors per value
        synthesized: Dict[str, tf.Tensor] = OrderedDict()
        for value, y in zip(self.values, ys):
            synthesized.update([(a, b.numpy())
                               for a, b in zip(value.learned_output_columns(), value.output_tensors(y=y))])

        for value in self.conditions:
            for name in value.learned_output_columns():
                synthesized[name] = cs[name]

        return synthesized

    @tf.function
    def _synthesize(self, n: tf.Tensor, cs: Dict[str, tf.Tensor], num_cs: tf.Tensor) -> List[tf.Tensor]:
        """Generate the given number of instances.

        Args:
            n: Number of instances to generate.
            cs: Condition tensor per column.

        Returns:
            Output tensor per column.

        """

        x = self.encoding.sample(n=n)

        if num_cs > tf.constant(0, dtype=tf.int64):
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=[cs[name] for name in value.learned_input_columns()])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            x = tf.concat(values=(x, c), axis=1)

        x = self.decoder(x)
        y = self.linear_output(x)

        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )
        return ys
