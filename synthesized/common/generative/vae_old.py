from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional

import tensorflow as tf

from .generative import Generative
from ..values import Value
from ..module import tensorflow_name_scoped, module_registry
from ..transformations import ResnetTransformation, DenseTransformation, Transformation
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
        distribution: str, latent_size: int,
        # Encoder and decoder network
        network: str, capacity: int, num_layers: int, residual_depths: Union[None, int, List[int]], batchnorm: bool,
        activation: str,
        # Optimizer
        optimizer: str, learning_rate: tf.Tensor, decay_steps: Optional[int], decay_rate: Optional[float],
        initial_boost: int, clip_gradients: float,
        # Beta KL loss coefficient
        beta: float,
        # Weight decay
        weight_decay: float,
        summarize: bool = False, summarize_gradient_norms: bool = False
    ):
        super().__init__(name=name, values=values, conditions=conditions)
        self.latent_size = latent_size
        self.beta = beta
        self.summarize = summarize
        self.summarize_gradient_norms = summarize_gradient_norms
        self._trainable_variables = None
        self.xs: Dict[str, tf.Tensor] = dict()

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
            self.encoder = module_registry[network].__init__(
                name='encoder',
                input_size=self.linear_input.size(), layer_sizes=[capacity for _ in range(num_layers)],
                weight_decay=weight_decay
            )
            if not isinstance(self.encoder, Transformation):
                raise ValueError

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

        self.optimizer = Optimizer(
            name='optimizer', optimizer=optimizer,
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
        if len(self.xs) == 0:
            return dict(), tf.no_op()

        x = self.unified_inputs(self.xs)

        #################################
        x = self.linear_input(x)
        x = self.encoder(x)
        z = self.encoding(x)
        x = self.add_conditions(z, conditions=self.xs)
        x = self.decoder(x)
        y = self.linear_output(x)
        #################################

        # Losses
        losses = self.value_losses(y=y, inputs=self.xs)
        encoding_loss = tf.identity(self.encoding.losses[0], name='encoding_loss')
        reconstruction_loss = tf.identity(losses['reconstruction-loss'], name='reconstruction_loss')
        regularization_loss = tf.identity(losses['regularization-loss'], name='regularization_loss')

        total_loss = tf.add_n(
            inputs=[encoding_loss, reconstruction_loss, regularization_loss], name='total_loss'
        )
        losses['encoding-loss'] = encoding_loss
        losses['total-loss'] = total_loss

        # Summaries
        for name, loss in losses.items():
            tf.summary.scalar(name=name, data=loss)

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
        self.optimizer.optimize(
            loss=self.loss, variables=self.get_trainable_variables
        )

        return

    @tf.function
    @tensorflow_name_scoped
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

        #################################
        x = self.unified_inputs(xs)
        x = self.linear_input.transform(x)
        x = self.encoder.transform(x=x)

        latent_space = self.encoding.encode(x=x)
        mean = self.encoding.mean.output
        std = self.encoding.stddev.output

        x = self.add_conditions(x=latent_space, conditions=xs)
        x = self.decoder.transform(x=x)
        y = self.linear_output.transform(x=x)
        synthesized = self.value_outputs(y=y, conditions=cs)
        #################################

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
        y = self._synthesize(n=n, cs=cs)
        synthesized = self.value_outputs(y=y, conditions=cs)

        return synthesized

    @tf.function
    def _synthesize(self, n: tf.Tensor, cs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Generate the given number of instances.

        Args:
            n: Number of instances to generate.
            cs: Condition tensor per column.

        Returns:
            Output tensor per column.

        """
        x = self.encoding.sample(n=n)
        x = self.add_conditions(x=x, conditions=cs)
        x = self.decoder(x)
        y = self.linear_output(x)

        return y

    @tensorflow_name_scoped
    def unified_inputs(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Concatenate input tensors per value
        x = tf.concat(values=[
            value.unify_inputs(xs=[inputs[name] for name in value.learned_input_columns()])
            for value in self.values if value.learned_input_size() > 0
        ], axis=1)

        return x

    @tensorflow_name_scoped
    def add_conditions(self, x: tf.Tensor, conditions: Dict[str, tf.Tensor]) -> tf.Tensor:
        if len(self.conditions) > 0:
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=[conditions[name] for name in value.learned_input_columns()])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            x = tf.concat(values=(x, c), axis=1)

        return x

    @tensorflow_name_scoped
    def value_losses(self, y: tf.Tensor, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )

        losses: Dict[str, tf.Tensor] = OrderedDict()

        # Reconstruction loss per value
        for value, y in zip(self.values, ys):
            losses[value.name + '-loss'] = value.loss(
                y=y, xs=[inputs[name] for name in value.learned_output_columns()]
            )

        # Regularization loss
        reg_losses = tf.compat.v1.losses.get_regularization_losses()
        if len(reg_losses) > 0:
            losses['regularization-loss'] = tf.add_n(inputs=reg_losses, name='regularization_loss')
        else:
            losses['regularization-loss'] = tf.constant(0, dtype=tf.float32)

        # Reconstruction loss
        losses['reconstruction-loss'] = tf.add_n(inputs=list(losses.values()), name='reconstruction_loss')

        return losses

    @tensorflow_name_scoped
    def value_outputs(self, y: tf.Tensor, conditions: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
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
                synthesized[name] = conditions[name]

        return synthesized

    def get_trainable_variables(self):
        if self._trainable_variables is None:
            self._trainable_variables = self.trainable_variables
        return self._trainable_variables
