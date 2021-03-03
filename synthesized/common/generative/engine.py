from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import tensorflow as tf

from .generative import Generative
from ..encodings import VariationalEncoding
from ..module import module_registry, tensorflow_name_scoped
from ..optimizers import Optimizer
from ..transformations import DenseTransformation
from ..values import DataFrameValue
from ...config import EngineConfig


class HighDimEngine(Generative):
    """Variational auto-encoder.

    The VAE consists of an NN-parametrized input-conditioned encoder distribution q(z|x), a latent
    prior distribution p'(z) and an NN-parametrized latent-conditioned decoder distribution p(y|z).
    The optimized loss consists of the reconstruction loss per value, the KL loss, and the regularization loss.
    The input and output are concatenated / split tensors per value. The encoder and decoder network use the same
    hyperparameters.
    """
    def __init__(self, name: str, df_value: DataFrameValue, config: EngineConfig = EngineConfig()):
        super(HighDimEngine, self).__init__(name=name, df_value=df_value)

        self.latent_size = config.latent_size
        self.network = config.network
        self.capacity = config.capacity
        self.num_layers = config.num_layers
        self.residual_depths = config.residual_depths
        self.batch_norm = config.batch_norm
        self.activation = config.activation
        self.optimizer_name = config.optimizer
        self.learning_rate = config.learning_rate
        self.decay_steps = config.decay_steps
        self.decay_rate = config.decay_rate
        self.initial_boost = config.initial_boost
        self.clip_gradients = config.clip_gradients
        self.beta = config.beta
        self.weight_decay = config.weight_decay
        self.l2 = tf.keras.regularizers.l2(config.weight_decay)

        self.linear_input = DenseTransformation(
            name='linear-input', input_size=self.df_value.learned_input_size(), output_size=self.capacity,
            batch_norm=False, activation='none'
        )

        kwargs = dict(
            name='encoder', input_size=self.linear_input.size(), depths=self.residual_depths,
            layer_sizes=[self.capacity for _ in range(self.num_layers)] if self.num_layers else None,
            output_size=self.capacity if not self.num_layers else None, activation=self.activation,
            batch_norm=self.batch_norm
        )
        for k in list(kwargs.keys()):
            if kwargs[k] is None:
                del kwargs[k]
        self.encoder = module_registry[self.network](**kwargs)

        self.encoding = VariationalEncoding(
            name='encoding',
            input_size=self.encoder.size(), encoding_size=self.latent_size, beta=self.beta
        )
        self.modulation = None

        kwargs['name'], kwargs['input_size'] = 'decoder', (self.encoding.size())
        self.decoder = module_registry[self.network](**kwargs)

        self.linear_output = DenseTransformation(
            name='linear-output',
            input_size=self.decoder.size(), output_size=self.df_value.learned_output_size(), batch_norm=False, activation='none'
        )

        self.optimizer = Optimizer(
            name='optimizer', optimizer=self.optimizer_name,
            learning_rate=self.learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate,
            clip_gradients=self.clip_gradients, initial_boost=self.initial_boost
        )

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            beta=self.beta, encoder=self.encoder.specification(),
            decoder=self.decoder.specification(), optimizer=self.optimizer.specification()
        )
        return spec

    @tensorflow_name_scoped
    def loss(self, xs: Dict[str, Sequence[tf.Tensor]]) -> tf.Tensor:
        if len(xs) == 0:
            return tf.constant(0, dtype=tf.float32)

        x = self.df_value.unify_inputs(xs)

        #################################
        x = self.linear_input(x)
        x = self.encoder(x)
        z = self.encoding(x)
        x = self.decoder(z)
        y = self.linear_output(x)
        #################################

        # Losses
        reconstruction_loss = tf.identity(
            self.df_value.loss(y=y, inputs=xs), name='reconstruction_loss')
        kl_loss = tf.identity(self.encoding.losses[0], name='kl_loss')

        with tf.name_scope("regularization"):
            regularization_loss = tf.add_n(
                inputs=[self.l2(w) for w in self.regularization_losses],
                name='regularization_loss'
            )

        total_loss = tf.add_n(
            inputs=[reconstruction_loss, kl_loss, regularization_loss], name='total_loss'
        )
        self.reconstruction_loss.assign(reconstruction_loss)
        self.regularization_loss.assign(regularization_loss)
        self.kl_loss.assign(kl_loss)
        self.total_loss.assign(total_loss)

        # Summaries
        tf.summary.scalar(name='reconstruction-loss', data=reconstruction_loss)
        tf.summary.scalar(name='kl-loss', data=kl_loss)
        tf.summary.scalar(name='regularization-loss', data=regularization_loss)
        tf.summary.scalar(name='total-loss', data=total_loss)

        return total_loss

    @tf.function
    def learn(self, xs: Dict[str, Sequence[tf.Tensor]]) -> None:
        """Training step for the generative model.

        Args:
            xs: Input tensor per column.

        Returns:
            Dictionary of loss tensors, and optimization operation.

        """

        with tf.GradientTape() as gg:
            total_loss = self.loss(xs)

        with tf.name_scope("optimization"):
            gradients = gg.gradient(total_loss, self.trainable_variables)
            grads_and_vars = list(zip(gradients, self.trainable_variables))
            self.optimizer.optimize(grads_and_vars)

        return

    @tf.function
    @tensorflow_name_scoped
    def encode(self, xs: Dict[str, Sequence[tf.Tensor]]) -> Tuple[Dict[str, tf.Tensor], Dict[str, Sequence[tf.Tensor]]]:
        """Encoding Step for VAE.

        Args:
            xs: Input tensor per column.

        Returns:
            Dictionary of Latent space tensor, means and stddevs, dictionary of output tensors per column

        """
        if len(xs) == 0:
            return tf.no_op(), dict()

        x = self.df_value.unify_inputs(xs)
        latent_space, mean, std, y = self._encode(x=x)
        synthesized = self.df_value.output_tensors(y=y)

        return {"sample": latent_space, "mean": mean, "std": std}, synthesized

    @tensorflow_name_scoped
    def _encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Encoding Step for VAE.

        Args:
            x: Input tensor per column.

        Returns:
            TF tensors with Latent space, means, stddevs and outputs

        """
        x = self.linear_input(x)
        x = self.encoder(x)

        mean = self.encoding.gaussian.mean(x)
        std = self.encoding.gaussian.stddev(x)
        latent_space = mean + std * tf.random.normal(shape=tf.shape(mean))

        x = self.decoder(latent_space)
        y = self.linear_output(x)

        return latent_space, mean, std, y

    @tensorflow_name_scoped
    def encode_deterministic(
            self, xs: Dict[str, Sequence[tf.Tensor]]
    ) -> Dict[str, Sequence[tf.Tensor]]:
        """Deterministic encoding for VAE.

        Args:
            xs: Input tensor per column.

        Returns:
            Dictionary of output tensors per column

        """
        if len(xs) == 0:
            return dict()

        x = self.df_value.unify_inputs(xs)
        y = self._encode_deterministic(x=x)
        synthesized = self.df_value.output_tensors(y=y, sample=False)

        return synthesized

    @tf.function
    @tensorflow_name_scoped
    def _encode_deterministic(self, x: tf.Tensor) -> tf.Tensor:
        """Encoding Step for VAE.

        Args:
            x: Input tensor per column.

        Returns:
            TF tensors with Latent space, means, stddevs and outputs

        """
        x = self.linear_input(x)
        x = self.encoder(x)

        mean = self.encoding.gaussian.mean(x)

        x = self.decoder(mean)
        y = self.linear_output(x)

        return y

    @tf.function
    @tensorflow_name_scoped
    def synthesize(self, n: tf.Tensor) -> Dict[str, Sequence[tf.Tensor]]:
        """Generate the given number of instances.

        Args:
            n: Number of instances to generate.

        Returns:
            Output tensor per column.

        """
        y = self._synthesize(n=n)
        synthesized = self.df_value.output_tensors(y=y)

        return synthesized

    def _synthesize(self, n: tf.Tensor) -> tf.Tensor:
        """Generate the given number of instances.

        Args:
            n: Number of instances to generate.

        Returns:
            Output tensor per column.

        """
        x = self.encoding.sample(n)
        x = self.decoder(x)
        y = self.linear_output(x)

        return y

    @property
    def regularization_losses(self):
        return [
            loss
            for module in [
                self.linear_input, self.encoder, self.encoding, self.decoder, self.linear_output
            ] + list(self.df_value.values())
            for loss in module.regularization_losses
        ]

    def get_variables(self) -> Dict[str, Any]:
        variables = super().get_variables()
        variables.update(
            latent_size=self.latent_size,
            beta=self.beta,
            weight_decay=self.weight_decay,
            linear_input=self.linear_input.get_variables(),
            encoder=self.encoder.get_variables(),
            encoding=self.encoding.get_variables(),
            decoder=self.decoder.get_variables(),
            linear_output=self.linear_output.get_variables(),
            optimizer=self.optimizer.get_variables()
        )
        return variables

    def set_variables(self, variables: Dict[str, Any]):
        super().set_variables(variables)

        assert self.latent_size == variables['latent_size']

        self.beta = variables['beta']
        self.weight_decay = variables['weight_decay']

        self.linear_input.set_variables(variables['linear_input'])
        self.encoder.set_variables(variables['encoder'])
        self.encoding.set_variables(variables['encoding'])
        self.decoder.set_variables(variables['decoder'])
        self.linear_output.set_variables(variables['linear_output'])
        self.optimizer.set_variables(variables['optimizer'])
