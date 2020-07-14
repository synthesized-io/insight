from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, Any

import tensorflow as tf

from .generative import Generative
from ..encodings import VariationalLSTMEncoding, VariationalRecurrentEncoding, RecurrentDSSEncoding
from ..module import tensorflow_name_scoped, module_registry
from ..optimizers import Optimizer
from ..transformations import DenseTransformation
from ..values import Value, IdentifierValue, ValueOps


class SeriesVAE(Generative):
    def __init__(
            self, name: str, values: List[Value], conditions: List[Value],
            encoding: str, identifier_label: Optional[str], identifier_value: Optional[IdentifierValue],
            # Latent space
            latent_size: int,
            # Encoder and decoder network
            network: str, capacity: int, num_layers: int, residual_depths: Union[None, int, List[int]],
            batch_norm: bool, activation: str, series_dropout: Optional[float],
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
        self.network = network
        self.capacity = capacity
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.series_dropout = series_dropout
        self.encoding_size = capacity

        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay
        self.l2 = tf.keras.regularizers.l2(weight_decay)

        self.identifier_label = identifier_label
        self.identifier_value = identifier_value

        self.value_ops = ValueOps(values=values, conditions=conditions, identifier=identifier_value)

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

        kwargs['name'], kwargs['input_size'] = 'decoder', self.encoding_size
        self.decoder = module_registry[network](**kwargs)

        self.linear_output = DenseTransformation(
            name='linear-output',
            input_size=self.decoder.size(), output_size=self.value_ops.output_size, batch_norm=False, activation='none'
        )

        if encoding == 'lstm':
            self.encoding = VariationalLSTMEncoding(
                name='encoding',
                input_size=self.encoder.size(), encoding_size=self.encoding_size,
                beta=self.beta
            )

        elif encoding == 'vrae':
            self.encoding = VariationalRecurrentEncoding(
                name='encoding',
                input_size=self.encoder.size(), encoding_size=self.encoding_size,
                beta=self.beta
            )

        elif encoding == 'rdssm':
            def emission_function(z: tf.Tensor):
                y = self.decoder(inputs=z)
                return y

            self.encoding = RecurrentDSSEncoding(
                name='encoding', input_size=self.encoder.size(), encoding_size=self.encoding_size, beta=self.beta,
                emission_function=emission_function
            )

        self.optimizer = Optimizer(
            name='optimizer', optimizer=optimizer,
            learning_rate=self.learning_rate, clip_gradients=clip_gradients,
            decay_steps=self.decay_steps, decay_rate=self.decay_rate,
            initial_boost=initial_boost
        )

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            beta=self.beta, encoder=self.encoder.specification(), lstm_mode=self.lstm_mode,
            decoder=self.decoder.specification(), optimizer=self.optimizer.specification()
        )
        return spec

    def loss(self):
        if len(self.xs) == 0:
            return dict(), tf.no_op()

        x = self.value_ops.unified_inputs(self.xs)
        if self.identifier_label and self.identifier_value:
            identifier = self.identifier_value.unify_inputs(
                xs=[tensor[:, 0] for tensor in self.xs[self.identifier_label]]
            )
        else:
            identifier = None

        #################################
        x = self.linear_input(x)
        x = self.encoder(x)
        x = self.value_ops.add_conditions(x, conditions=self.xs)
        x = self.encoding(x, identifier=identifier, series_dropout=self.series_dropout)
        x = self.decoder(x)
        y = self.linear_output(x)
        #################################

        # Losses
        self.losses: Dict[str, tf.Tensor] = OrderedDict()

        reconstruction_loss = tf.identity(
            self.value_ops.reconstruction_loss(y, self.xs), name='reconstruction_loss')
        kl_loss = tf.identity(self.encoding.losses[0], name='kl_loss')
        regularization_loss = tf.add_n(
            inputs=[self.l2(w) for w in self.regularization_losses],
            name='regularization_loss'
        )

        total_loss = tf.add_n([kl_loss, reconstruction_loss, regularization_loss], name='total_loss')

        self.losses['reconstruction-loss'] = reconstruction_loss
        self.losses['kl-loss'] = kl_loss
        self.losses['regularization-loss'] = regularization_loss
        self.losses['total-loss'] = total_loss

        # Summaries
        tf.summary.scalar(name='reconstruction-loss', data=reconstruction_loss)
        tf.summary.scalar(name='kl-loss', data=kl_loss)
        tf.summary.scalar(name='regularization-loss', data=regularization_loss)
        tf.summary.scalar(name='total-loss', data=total_loss)

        return total_loss

    @tf.function
    @tensorflow_name_scoped
    def learn(self, xs: Dict[str, List[tf.Tensor]]) -> None:
        self.xs = xs
        # Optimization step
        self.optimizer.optimize(
            loss=self.loss, variables=self.get_trainable_variables
        )

        return

    def encode(self, xs: Dict[str, List[tf.Tensor]], cs: Dict[str, List[tf.Tensor]], n_forecast: int = 0) -> \
            Tuple[Dict[str, tf.Tensor], Dict[str, List[tf.Tensor]]]:
        if len(self.xs) == 0:
            return dict(), tf.no_op()

        x = self.value_ops.unified_inputs(xs)
        x = tf.expand_dims(x, axis=0)

        # Get identifier
        if self.identifier_label and self.identifier_value:
            identifier = self.identifier_value.unify_inputs(xs=[xs[self.identifier_label][0]])
            identifier = tf.expand_dims(identifier, axis=0)
        else:
            identifier = None

        latent_space, mean, std, y = self._encode(x, cs, identifier, tf.constant(n_forecast))
        synthesized = self.value_ops.value_outputs(y, conditions=cs)

        return {"sample": latent_space, "mean": mean, "std": std}, synthesized

    @tf.function
    def _encode(self, x: tf.Tensor, cs: Dict[str, List[tf.Tensor]], identifier: Optional[tf.Tensor],
                n_forecast: tf.Tensor = 0) -> Tuple[tf.Tensor, ...]:

        x = self.linear_input(x)
        x = self.encoder(x)
        x = self.value_ops.add_conditions(x, conditions=cs)
        x, latent_space = self.encoding(x, identifier=identifier, return_encoding=True, n_forecast=n_forecast)
        mean = self.encoding.mean.output
        std = self.encoding.stddev.output
        x = self.decoder(x)
        y = self.linear_output(x)
        # Remove third dimension, as we synthesize one series per step
        y = tf.squeeze(y, axis=0)
        return latent_space, mean, std, y

    def synthesize(self, n: int, cs: Dict[str, List[tf.Tensor]],
                   identifier: tf.Tensor = None) -> Dict[str, List[tf.Tensor]]:
        y, identifier = self._synthesize(n=n, cs=cs, identifier=identifier)
        synthesized = self.value_ops.value_outputs(y, conditions=cs, identifier=identifier)

        return synthesized

    # @tf.function
    def _synthesize(
            self, n: int, cs: Dict[str, List[tf.Tensor]], identifier: tf.Tensor = None
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:

        if self.identifier_value is not None:
            if identifier is not None:
                identifier_embedding = self.identifier_value.get_embedding(identifier)
            else:
                identifier, identifier_embedding = self.identifier_value.random_value_from_normal()
            identifier_embedding = tf.squeeze(identifier_embedding)
            identifier = tf.tile(identifier, multiples=(n,))
        else:
            identifier, identifier_embedding = None, None

        x = self.encoding.sample(n=n, identifier=identifier_embedding)
        x = self.value_ops.add_conditions(x=x, conditions=cs)
        x = self.decoder(x)
        y = self.linear_output(x)
        # Remove third dimension, as we synthesize one series per step
        y = tf.squeeze(y, axis=0)
        return y, identifier

    @property
    def regularization_losses(self):
        return [
            loss
            for module in [self.linear_input, self.encoder, self.encoding, self.decoder, self.linear_output]+self.values
            for loss in module.regularization_losses
        ]

    def get_variables(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_variables(self, variables: Dict[str, Any]):
        raise NotImplementedError
