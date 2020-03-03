from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional

import tensorflow as tf

from .generative import Generative
from ..module import tensorflow_name_scoped, module_registry
from ..transformations import DenseTransformation
from ..encodings import VariationalLSTMEncoding, VariationalRecurrentEncoding, RecurrentDSSEncoding
from ..values import Value, IdentifierValue, ValueOps
from ..optimizers import Optimizer


class SeriesVAE(Generative):
    def __init__(
            self, name: str, values: List[Value], conditions: List[Value],
            encoding: str, identifier_label: Optional[str], identifier_value: Optional[IdentifierValue],
            # Latent space
            latent_size: int,
            # Encoder and decoder network
            network: str, capacity: int, num_layers: int, residual_depths: Union[None, int, List[int]],
            batchnorm: bool, activation: str, series_dropout: Optional[float],
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
        self.batchnorm = batchnorm
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
            input_size=self.value_ops.input_size, output_size=capacity, batchnorm=False, activation='none'
        )

        kwargs = dict(
            name='encoder', input_size=self.linear_input.size(), depths=residual_depths,
            layer_sizes=[capacity for _ in range(num_layers)] if num_layers else None,
            output_size=capacity if not num_layers else None, activation=activation, batchnorm=batchnorm
        )
        for k in list(kwargs.keys()):
            if kwargs[k] is None:
                del kwargs[k]
        self.encoder = module_registry[network](**kwargs)

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

        kwargs['name'], kwargs['input_size'] = 'decoder', self.encoding_size
        self.decoder = module_registry[network](**kwargs)

        self.linear_output = DenseTransformation(
            name='linear-output',
            input_size=self.decoder.size(), output_size=self.value_ops.output_size, batchnorm=False, activation='none'
        )

        if encoding == 'rdssm':
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
            identifier = self.identifier_value.unify_inputs(xs=[self.xs[self.identifier_label][:, 0]])
        else:
            identifier = None

        #################################
        x = self.linear_input(inputs=x)
        x = self.encoder(inputs=x)
        x = self.value_ops.add_conditions(x, conditions=self.xs)
        x = self.encoding(inputs=x, identifier=identifier, series_dropout=self.series_dropout)
        x = self.decoder(inputs=x)
        y = self.linear_output(inputs=x)
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

        total_loss = tf.add_n(inputs=[kl_loss, reconstruction_loss, regularization_loss], name='total_loss')

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
    def learn(self, xs: Dict[str, tf.Tensor]) -> None:
        self.xs = xs
        # Optimization step
        self.optimizer.optimize(
            loss=self.loss, variables=self.get_trainable_variables
        )

        return

    @tf.function
    def encode(self, xs: Dict[str, tf.Tensor], cs: Dict[str, tf.Tensor]) -> \
            Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        if len(self.xs) == 0:
            return dict(), tf.no_op()

        x = self.value_ops.unified_inputs(xs)
        if self.identifier_label and self.identifier_value:
            identifier = self.identifier_value.unify_inputs(xs=[xs[self.identifier_label][0]])
        else:
            identifier = None

        #################################
        x = self.linear_input(inputs=x)
        x = self.encoder(inputs=x)
        x = self.value_ops.add_conditions(x, conditions=cs)
        x, latent_space = self.encoding(inputs=x, identifier=identifier, return_encoding=True)
        mean = self.encoding.mean.output
        std = self.encoding.stddev.output
        x = self.decoder(inputs=x)
        y = self.linear_output(inputs=x)
        synthesized = self.value_ops.value_outputs(y=y, conditions=cs)
        #################################

        return {"sample": latent_space, "mean": mean, "std": std}, synthesized

    def synthesize(self, n: int, cs: Dict[str, tf.Tensor], identifier: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        y, identifier = self._synthesize(n=n, cs=cs, identifier=identifier)
        synthesized = self.value_ops.value_outputs(y=y, conditions=cs, identifier=identifier)

        return synthesized

    def _synthesize(
            self, n: int, cs: Dict[str, tf.Tensor], identifier: tf.Tensor = None
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:

        if self.identifier_value is not None:
            if identifier is not None:
                identifier_embedding = self.identifier_value.get_embedding(identifier)
            else:
                identifier, identifier_embedding = self.identifier_value.random_value_from_normal()
            identifier_embedding = tf.squeeze(identifier_embedding)
            identifier = tf.tile(input=identifier, multiples=(n,))
        else:
            identifier, identifier_embedding = None, None

        x = self.encoding.sample(n=n, identifier=identifier_embedding)
        x = self.value_ops.add_conditions(x=x, conditions=cs)
        x = self.decoder(inputs=x)
        y = self.linear_output(inputs=x)

        return y, identifier

    @property
    def regularization_losses(self):
        return [
            loss
            for module in [self.linear_input, self.encoder, self.encoding, self.decoder, self.linear_output]+self.values
            for loss in module.regularization_losses
        ]
