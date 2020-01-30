from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import tensorflow as tf

from .generative import Generative
from ..module import tensorflow_name_scoped
from ..values import Value, IdentifierValue


class SeriesVAE(Generative):
    def __init__(
            self, name: str, values: List[Value], conditions: List[Value],
            lstm_mode: int, identifier_label: str, identifier_value: IdentifierValue,
            # Latent space
            latent_size: int,
            # Encoder and decoder network
            network: str, capacity: int, num_layers: int, residual_depths: Union[None, int, List[int]],
            batchnorm: bool, activation: str,
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
        self.lstm_mode = lstm_mode

        self.latent_size = latent_size
        self.beta = beta
        self.summarize = summarize
        self.network = network
        self.capacity = capacity
        self.num_layers = num_layers
        self.batchnorm = batchnorm
        self.activation = activation
        self.encoding_size = capacity

        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay

        self.identifier_label = identifier_label
        self.identifier_value = identifier_value

        self.encoding_type = 'variational'

        # Total input and output size of all values
        input_size = 0
        output_size = 0
        for value in self.values:
            input_size += value.learned_input_size()
            output_size += value.learned_output_size()

        self.linear_input = self.add_module(
            module='dense', name='linear-input',
            input_size=input_size, output_size=capacity, batchnorm=False, activation='none'
        )

        if network == 'resnet':
            self.encoder = self.add_module(
                module=network, name='encoder',
                input_size=self.linear_input.size(), depths=residual_depths,
                layer_sizes=[capacity for _ in range(num_layers)], weight_decay=weight_decay
            )
        else:
            self.encoder = self.add_module(
                module=network, name='encoder',
                input_size=self.linear_input.size(),
                layer_sizes=[capacity for _ in range(num_layers)], weight_decay=weight_decay
            )

        if self.lstm_mode == 1:
            self.encoding = self.add_module(
                module='variational', name='encoding',
                input_size=self.encoder.size(), encoding_size=self.encoding_size,
                beta=self.beta
            )

            self.lstm = self.add_module(
                module='lstm', name='lstm',
                input_size=self.encoding.size(), output_size=self.capacity
            )

            decoder_input_size = self.lstm.size()

        else:
            self.lstm = None

            self.encoding = self.add_module(
                module='rnn_variational', name='encoding',
                input_size=self.encoder.size(), encoding_size=self.encoding_size,
                beta=self.beta
            )

            decoder_input_size = self.encoding.size()

        if network == 'resnet':
            self.decoder = self.add_module(
                module=network, name='decoder',
                input_size=decoder_input_size, depths=residual_depths,
                layer_sizes=[capacity for _ in range(num_layers)], weight_decay=weight_decay
            )
        else:
            self.decoder = self.add_module(
                module=network, name='decoder',
                input_size=decoder_input_size,
                layer_sizes=[capacity for _ in range(num_layers)], weight_decay=weight_decay
            )

        self.linear_output = self.add_module(
            module='dense', name='linear-output',
            input_size=self.decoder.size(), output_size=output_size, batchnorm=False, activation='none'
        )

        self.optimizer = self.add_module(
            module='optimizer', name='optimizer', optimizer=optimizer,
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

    @tensorflow_name_scoped
    def learn(self, xs: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Operation]:

        if len(xs) == 0:
            return dict(), tf.no_op()

        losses: Dict[str, tf.Tensor] = OrderedDict()
        summaries = list()

        # Concatenate input tensors per value
        x = tf.concat(values=[
            value.unify_inputs(xs=[xs[name] for name in value.learned_input_columns()])
            for value in self.values if value.learned_input_size() > 0
        ], axis=1)

        # Get identifier
        if self.identifier_label:
            identifier = self.identifier_value.unify_inputs(xs=[xs[self.identifier_label][0]])

        #################################

        x = self.linear_input.transform(x=x)
        x = self.encoder.transform(x=x)

        if len(self.conditions) > 0:
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=[xs[name] for name in value.learned_input_columns()])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            x = tf.concat(values=(x, c), axis=1)

        if self.lstm_mode == 1:
            encoding, encoding_loss, mean, stddev = self.encoding.encode(x=x)

            if self.identifier_label is None:
                x = self.lstm.transform(x=encoding)
            else:
                x = self.lstm.transform(x=encoding, state=identifier)
        else:
            x, encoding, encoding_loss, mean, stddev = self.encoding.encode(
                x=x, encoding_plus_loss=True
            )

        if self.summarize:
            summaries.append(tf.contrib.summary.scalar(
                name='encoding-mean', tensor=mean, family=None, step=None
            ))
            summaries.append(tf.contrib.summary.scalar(
                name='encoding-variance', tensor=stddev, family=None, step=None
            ))
            summaries.append(tf.contrib.summary.scalar(
                name='encoding-loss', tensor=encoding_loss, family=None, step=None
            ))

        x = self.decoder.transform(x=x)
        y = self.linear_output.transform(x=x)

        #################################

        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )

        losses['encoding-loss'] = encoding_loss
        reconstruction_loss = 0

        # Reconstruction loss per value
        for value, y_ in zip(self.values, ys):
            value_loss = value.loss(
                y=y_, xs=[xs[name] for name in value.learned_output_columns()]
            )
            losses[value.name + '-loss'] = value_loss
            reconstruction_loss += value_loss

        # Reconstruction loss
        summaries.append(tf.contrib.summary.scalar(name='reconstruction-loss', tensor=reconstruction_loss))

        # Regularization loss
        reg_losses = tf.compat.v1.losses.get_regularization_losses()
        if len(reg_losses) > 0:
            losses['regularization-loss'] = tf.add_n(inputs=reg_losses)

        # Total loss
        total_loss = tf.add_n(inputs=list(losses.values()))
        losses['total-loss'] = total_loss

        # Loss summaries
        if self.summarize:
            for name, loss in losses.items():
                summaries.append(tf.contrib.summary.scalar(name=name, tensor=loss))
                if name not in ('total-loss', 'encoding-loss'):
                    summaries.append(tf.contrib.summary.scalar(
                        name=name + '-ratio', tensor=(loss / losses['encoding-loss'])
                    ))

        # Optimization step
        optimized = self.optimizer.optimize(
            loss=total_loss, summarize_gradient_norms=self.summarize, summaries=summaries
        )

        return losses, optimized

    @tensorflow_name_scoped
    def synthesize(self, n: tf.Tensor, cs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:

        x = self.encoding.sample(n=n)

        if len(self.conditions) > 0:
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=[cs[name] for name in value.learned_input_columns()])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            x = tf.concat(values=(x, c), axis=1)

        if self.lstm_mode == 2 and self.identifier_label is not None:
            identifier = self.identifier_value.next_identifier()
            identifier = tf.tile(input=identifier, multiples=(n,))

        elif self.lstm_mode == 1:
            if self.identifier_label is None:
                x = self.lstm.transform(x=x)
            else:
                identifier, state = self.identifier_value.next_identifier_embedding()
                x = self.lstm.transform(x=x, state=state)

        x = self.decoder.transform(x=x)
        y = self.linear_output.transform(x=x)
        ys = tf.split(value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
                      axis=1, num=None, name=None)

        # Output tensors per value
        synthesized = OrderedDict()
        if self.identifier_value is not None:
            synthesized[self.identifier_label] = identifier

        for value, y in zip(self.values, ys):
            synthesized.update(zip(value.learned_output_columns(), value.output_tensors(y=y)))

        for value in self.conditions:
            for name in value.learned_output_columns():
                synthesized[name] = cs[name]

        return synthesized
