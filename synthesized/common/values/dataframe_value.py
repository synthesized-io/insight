from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .associated_categorical import output_association_tensors
from .value import Value
from ..module import tensorflow_name_scoped
from ..rules import Association


class DataFrameValue(Value, Mapping[str, Value]):
    """
    Container for all values in a dataframe, applying value methods across all values
    also manages NaN values when specified by a value meta.

    Attributes:
        name: string name of value (altered in super().__init__() call)
        values: a dictionary containing all values
        identifier_value: (unused) tbd on whether this should be kept
    """
    def __init__(self, name: str, values: Dict[str, Value]):
        super(DataFrameValue, self).__init__(name=name)

        self._values: Dict[str, Value] = OrderedDict(**values)
        self.identifier_value = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

    def __getitem__(self, k: str) -> Value:
        return self._values[k]

    def __iter__(self) -> Iterator[str]:
        for key in self._values:
            yield key

    def __len__(self) -> int:
        return len(self._values)

    def columns(self) -> List[str]:
        return [column for value in self.values() for column in value.columns()]

    def learned_input_size(self) -> int:
        return sum(value.learned_input_size() for value in self.values())

    def learned_output_size(self) -> int:
        return sum(value.learned_output_size() for value in self.values())

    @tensorflow_name_scoped
    def unify_inputs(self, inputs: Dict[str, Sequence[tf.Tensor]]) -> tf.Tensor:
        """ Concatenate input tensors per value """
        values = []
        for name, value in self.items():
            if value.learned_input_size() > 0:
                values.append(
                    value.unify_inputs(xs=inputs[name])
                )
        x = tf.concat(values=values, axis=-1)

        return x

    def split_outputs(self, outputs: Dict[str, Sequence[Any]]) -> Dict[str, np.ndarray]:
        output_dict = dict()
        for name, value in self.items():
            output_dict.update(value.split_outputs({name: outputs[name]}))
        return output_dict

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, identifier: tf.Tensor = None,
                       sample: bool = True, association_rules: Sequence[Association] = None) -> Dict[str, Sequence[tf.Tensor]]:
        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values()],
            axis=-1
        )
        y_dict = {name: y for name, y in zip(self, ys)}

        # Output tensors per value
        synthesized: Dict[str, Sequence[tf.Tensor]] = {}
        association_rules = association_rules or []

        for association_rule in association_rules:
            synthesized.update(output_association_tensors(association_rule, y_dict))

        if identifier is not None and self.identifier_label is not None:
            synthesized[self.identifier_label] = (identifier,)

        for name, y in y_dict.items():
            if name in synthesized:  # if this is true then we've generated the outputs as part of an association
                continue
            synthesized[name] = self[name].output_tensors(y=y, sample=sample)

        return synthesized

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, inputs: Dict[str, Sequence[tf.Tensor]]) -> tf.Tensor:
        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values()],
            axis=-1
        )

        losses: Dict[str, tf.Tensor] = OrderedDict()

        # Reconstruction loss per value
        for (name, value), y in zip(self.items(), ys):
            losses[name + '-loss'] = value.loss(
                y=y, xs=inputs[name],
            )

        # Reconstruction loss
        reconstruction_loss = tf.add_n(inputs=list(losses.values()), name='reconstruction_loss')

        return reconstruction_loss
