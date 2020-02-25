from collections import OrderedDict
from typing import Dict, List, Union, Tuple, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from .generative import Generative
from ..values import Value, ValueOps
from ..module import tensorflow_name_scoped, module_registry
from ..distributions import Distribution
from ..optimizers import Optimizer


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
        network: str, capacity: int, num_layers: int, residual_depths: Union[None, int, List[int]], batchnorm: bool,
        activation: str,
        # Optimizer
        optimizer: str, learning_rate: tf.Tensor, decay_steps: Optional[int], decay_rate: Optional[float],
        initial_boost: int, clip_gradients: float,
        # Beta KL loss coefficient
        beta: float,
        # Weight decay
        weight_decay: float
    ):
        super().__init__(name=name, values=values, conditions=conditions)
        self.latent_size = latent_size
        self.beta = beta
        self.weight_decay = weight_decay
        self.l2 = tf.keras.regularizers.l2(weight_decay)

        # Total input and output size of all values
        self.value_ops = ValueOps(values=values, conditions=conditions)

        kwargs = dict(
            name='encoder', input_size=self.value_ops.input_size, depths=residual_depths,
            layer_sizes=[capacity for _ in range(num_layers)] if num_layers else None,
            output_size=capacity if not num_layers else None, activation=activation, batchnorm=batchnorm
        )
        for k in list(kwargs.keys()):
            if kwargs[k] is None:
                del kwargs[k]
        self.encoder = module_registry[network](**kwargs)

        self.encoding = Distribution(
            name='encoding', input_size=capacity, output_size=latent_size,
            distribution=distribution, beta=beta, encode=True
        )

        # Decoder: parametrized distribution p(y|z,c)
        kwargs['name'], kwargs['input_size'] = 'decoder', (self.encoding.size() + self.value_ops.condition_size)
        self.decoder = module_registry[network](**kwargs)

        self.decoding = Distribution(
            name='decoding',
            input_size=self.decoder.size(), output_size=self.value_ops.output_size,
            distribution='deterministic', beta=beta, encode=False
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

        x = self.value_ops.unified_inputs(self.xs)

        #################################
        x = self.encoder(x)
        q = self.encoding(x)
        z = q.sample()
        x = self.value_ops.add_conditions(x=z, conditions=self.xs)
        x = self.decoder(x)
        p = self.decoding(x)
        y = p.sample()
        #################################

        # Losses
        self.losses: Dict[str, tf.Tensor] = OrderedDict()

        reconstruction_loss = tf.identity(
            self.value_ops.reconstruction_loss(y=y, inputs=self.xs), name='reconstruction_loss')
        kl_loss = tf.identity(self.encoding.losses[0], name='kl_loss')
        reconstruction_loss = tf.identity(self.losses['reconstruction-loss'], name='reconstruction_loss')
        regularization_loss = tf.add_n(
            inputs=[self.l2(w) for w in self.regularization_losses],
            name='regularization_loss'
        )

        total_loss = tf.add_n(
            inputs=[kl_loss, reconstruction_loss, regularization_loss], name='total_loss'
        )
        self.losses['regularization-loss'] = regularization_loss
        self.losses['kl-loss'] = kl_loss
        self.losses['total-loss'] = total_loss

        # Summaries
        tf.summary.scalar(name='total-loss', data=total_loss)
        tf.summary.scalar(name='regularization-loss', data=regularization_loss)
        tf.summary.image(
            name='latent_space_correlation',
            data=tf.reshape(tf.abs(tfp.stats.correlation(z)), shape=(1, self.latent_size, self.latent_size, 1))
        )
        tf.summary.histogram(name='posterior', data=z)

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
        x = self.value_ops.unified_inputs(xs)
        x = self.encoder(x)
        q = self.encoding(x)

        latent_space = q.sample()
        mean = self.encoding.mean.output
        std = self.encoding.stddev.output

        x = self.value_ops.add_conditions(x=latent_space, conditions=cs)
        x = self.decoder(x)
        p = self.decoding(x)
        y = p.sample()
        synthesized = self.value_ops.value_outputs(y=y, conditions=cs)
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
        synthesized = self.value_ops.value_outputs(y=y, conditions=cs)

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
        prior = self.encoding.prior()
        z = prior.sample(sample_shape=(n,))
        z = self.value_ops.add_conditions(x=z, conditions=cs)
        x = self.decoder(z)
        p = self.decoding(x)
        y = p.sample()

        return y

    @property
    def regularization_losses(self):
        return [
            loss
            for module in [self.encoder, self.encoding, self.decoder, self.decoding] + self.values
            for loss in module.regularization_losses
        ]
