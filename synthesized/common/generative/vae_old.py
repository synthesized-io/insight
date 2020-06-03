from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, Any

import tensorflow as tf

from .generative import Generative
from ..encodings import VariationalEncoding
from ..module import tensorflow_name_scoped, module_registry
from ..optimizers import Optimizer
from ..transformations import DenseTransformation
from ..values import Value, ValueOps


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
        latent_size: int,
        # Encoder and decoder network
        network: str, capacity: int, num_layers: int, residual_depths: Union[None, int, List[int]], batch_norm: bool,
        activation: str,
        # Optimizer
        optimizer: str, learning_rate: float, decay_steps: Optional[int], decay_rate: Optional[float],
        initial_boost: int, clip_gradients: float,
        # Beta KL loss coefficient
        beta: float,
        # Weight decay
        weight_decay: float
    ):
        super(VAEOld, self).__init__(name=name, values=values, conditions=conditions)

        self.latent_size = latent_size
        self.network = network
        self.capacity = capacity
        self.num_layers = num_layers
        self.residual_depths = residual_depths
        self.batch_norm = batch_norm
        self.activation = activation
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.initial_boost = initial_boost
        self.clip_gradients = clip_gradients
        self.beta = beta
        self.weight_decay = weight_decay
        self.l2 = tf.keras.regularizers.l2(weight_decay)

        self.value_ops = ValueOps(values=values, conditions=conditions)

        self.linear_input = DenseTransformation(
            name='linear-input',
            input_size=self.value_ops.input_size, output_size=capacity, batch_norm=False, activation='none'
        )

        kwargs = dict(
            name='encoder', input_size=self.linear_input.size(), depths=residual_depths,
            layer_sizes=[capacity for _ in range(num_layers)] if num_layers else None,
            output_size=capacity if not num_layers else None, activation=activation, batch_norm=batch_norm
        )
        for k in list(kwargs.keys()):
            if kwargs[k] is None:
                del kwargs[k]
        self.encoder = module_registry[network](**kwargs)

        self.encoding = VariationalEncoding(
            name='encoding',
            input_size=self.encoder.size(), encoding_size=self.latent_size, beta=beta
        )

        self.modulation = None

        kwargs['name'], kwargs['input_size'] = 'decoder', (self.encoding.size() + self.value_ops.condition_size)
        self.decoder = module_registry[network](**kwargs)

        self.linear_output = DenseTransformation(
            name='linear-output',
            input_size=self.decoder.size(), output_size=self.value_ops.output_size, batch_norm=False, activation='none'
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
        x = self.linear_input(x)
        x = self.encoder(x)
        z = self.encoding(x)
        x = self.value_ops.add_conditions(z, conditions=self.xs)
        x = self.decoder(x)
        y = self.linear_output(x)
        #################################

        # Losses
        self.losses: Dict[str, tf.Tensor] = OrderedDict()

        reconstruction_loss = tf.identity(
            self.value_ops.reconstruction_loss(y=y, inputs=self.xs), name='reconstruction_loss')
        kl_loss = tf.identity(self.encoding.losses[0], name='kl_loss')
        regularization_loss = tf.add_n(
            inputs=[self.l2(w) for w in self.regularization_losses],
            name='regularization_loss'
        )

        total_loss = tf.add_n(
            inputs=[reconstruction_loss, kl_loss, regularization_loss], name='total_loss'
        )
        self.losses['reconstruction-loss'] = reconstruction_loss
        self.losses['regularization-loss'] = regularization_loss
        self.losses['kl-loss'] = kl_loss
        self.losses['total-loss'] = total_loss

        # Summaries
        tf.summary.scalar(name='reconstruction-loss', data=reconstruction_loss)
        tf.summary.scalar(name='kl-loss', data=kl_loss)
        tf.summary.scalar(name='regularization-loss', data=regularization_loss)
        tf.summary.scalar(name='total-loss', data=total_loss)

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
    def encode(self, xs: Dict[str, tf.Tensor],
               cs: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
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
        x = self.linear_input(x)
        x = self.encoder(x)

        latent_space = self.encoding(x)
        mean = self.encoding.gaussian.mean.output
        std = self.encoding.gaussian.stddev.output

        x = self.value_ops.add_conditions(x=latent_space, conditions=cs)
        x = self.decoder(x)
        y = self.linear_output(x)
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
        y = self._synthesize(n=tf.constant(n, dtype=tf.int64), cs=cs)
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
        x = self.encoding.sample(n)
        x = self.value_ops.add_conditions(x=x, conditions=cs)
        x = self.decoder(x)
        y = self.linear_output(x)

        return y

    @property
    def regularization_losses(self):
        return [
            loss
            for module in [self.linear_input, self.encoder, self.encoding, self.decoder, self.linear_output]+self.values
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
