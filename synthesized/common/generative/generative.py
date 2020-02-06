from collections import OrderedDict
from typing import Dict, List, Tuple

import tensorflow as tf

from ..module import tensorflow_name_scoped
from ..values import Value


class Generative(tf.Module):
    """Base class for generative models."""

    def __init__(self, name: str, values: List[Value], conditions: List[Value]):
        super(Generative, self).__init__(name=name)

        self.values = values
        self.conditions = conditions
        self._trainable_variables = None
        self.xs: Dict[str, tf.Tensor] = dict()
        self.losses: Dict[str, tf.Tensor] = dict()

    @tensorflow_name_scoped
    def learn(self, xs: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Operation]:
        """Training step for the generative model.

        Args:
            xs: Input tensor per column.

        Returns:
            Dictionary of loss tensors, and optimization operation.

        """
        raise NotImplementedError

    @tensorflow_name_scoped
    def synthesize(self, n: tf.Tensor, cs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Generate the given number of instances.

        Args:
            n: Number of instances to generate.
            cs: Condition tensor per column.

        Returns:
            Output tensor per column.

        """
        raise NotImplementedError

    def specification(self):
        spec = dict(name=self._name)
        return spec

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
