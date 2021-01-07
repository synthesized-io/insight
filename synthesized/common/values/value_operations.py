from collections import OrderedDict
from typing import Dict, List, Sequence

import tensorflow as tf

from .value import Value
from ..module import tensorflow_name_scoped


class ValueOps(tf.Module):
    """Layer that handles all of the Value related TensorFlow functions"""
    def __init__(self, values: List[Value], conditions: List[Value], identifier: Value = None, name: str = 'value_ops'):
        super(ValueOps, self).__init__(name=name)

        self.values = values
        self.conditions = conditions
        self.identifier_value = identifier
        self.identifier_label = None if identifier is None else identifier.name

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
            self.condition_size += value.learned_input_size()

    @tensorflow_name_scoped
    def unified_inputs(self, inputs: Dict[str, Sequence[tf.Tensor]]) -> tf.Tensor:
        # Concatenate input tensors per value
        x = tf.concat(values=[
            value.unify_inputs(xs=inputs[value.name])
            for value in self.values if value.learned_input_size() > 0
        ], axis=-1)

        return x

    @tensorflow_name_scoped
    def add_conditions(self, x: tf.Tensor, conditions: Dict[str, Sequence[tf.Tensor]]) -> tf.Tensor:
        if len(self.conditions) > 0:
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=conditions[value.name])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            x = tf.concat(values=(x, c), axis=1)

        return x

    @tensorflow_name_scoped
    def reconstruction_loss(self, y: tf.Tensor, inputs: Dict[str, Sequence[tf.Tensor]]) -> tf.Tensor:
        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=-1
        )

        losses: Dict[str, tf.Tensor] = OrderedDict()

        # Reconstruction loss per value
        for value, y in zip(self.values, ys):
            losses[value.name + '-loss'] = value.loss(
                y=y, xs=inputs[value.name]
            )

        # Reconstruction loss
        reconstruction_loss = tf.add_n(inputs=list(losses.values()), name='reconstruction_loss')

        return reconstruction_loss

    @tensorflow_name_scoped
    def value_outputs(self, y: tf.Tensor, conditions: Dict[str, Sequence[tf.Tensor]], identifier: tf.Tensor = None,
                      sample: bool = True, produce_nans: bool = False) -> Dict[str, Sequence[tf.Tensor]]:
        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=-1
        )

        # Output tensors per value
        synthesized: Dict[str, Sequence[tf.Tensor]] = OrderedDict()

        if identifier is not None and self.identifier_label is not None:
            synthesized[self.identifier_label] = (identifier,)

        for value, y in zip(self.values, ys):
            synthesized[value.name] = value.output_tensors(y=y, sample=sample, produce_nans=produce_nans)

        for value in self.conditions:
            synthesized[value.name] = conditions[value.name]

        return synthesized
