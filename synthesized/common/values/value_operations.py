from collections import OrderedDict
from typing import Dict, List

import tensorflow as tf

from synthesized.common.module import tensorflow_name_scoped
from synthesized.common.values import Value


class ValueOps(tf.Module):
    """Layer that handles all of the Value related TensorFlow functions"""
    def __init__(self, values: List[Value], conditions: List[Value], name: str = 'value_ops'):
        super(ValueOps, self).__init__(name=name)

        self.values = values
        self.conditions = conditions

        # Total input and output size of all values
        self.input_size = 0
        self.output_size = 0
        for value in self.values:
            self.input_size += value.learned_input_size()
            self.output_size += value.learned_output_size()

        # Total condition size
        self.condition_size = 0
        for value in self.conditions:
            assert value.learned_input_size() > 0
            assert value.learned_input_columns() == value.learned_output_columns()
            self.condition_size += value.learned_input_size()

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
    def reconstruction_loss(self, y: tf.Tensor, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
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
        reconstruction_loss = tf.add_n(inputs=list(losses.values()), name='reconstruction_loss')

        return reconstruction_loss

    @tensorflow_name_scoped
    def value_outputs(self, y: tf.Tensor, conditions: Dict[str, tf.Tensor],
                      identifier: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )

        # Output tensors per value
        synthesized: Dict[str, tf.Tensor] = OrderedDict()

        if identifier is not None:
            synthesized[self.identifier_label] = identifier

        for value, y in zip(self.values, ys):
            synthesized.update(zip(value.learned_output_columns(), value.output_tensors(y=y)))

        for value in self.conditions:
            for name in value.learned_output_columns():
                synthesized[name] = conditions[name]

        return synthesized
