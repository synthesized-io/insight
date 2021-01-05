from typing import List, Sequence

import tensorflow as tf

from ..module import tensorflow_name_scoped
from .value import Value


class RuleValue(Value):

    def __init__(self, name: str, values: List[Value], num_learned: int):
        super().__init__(name=name, meta_names=[name for value in values for name in value.meta_names])
        self.num_learned = num_learned
        self.values = values

    def __str__(self) -> str:
        return self.__class__.__name__[:-5].lower()

    def columns(self) -> List[str]:
        columns = list()
        for value in self.values:
            columns.extend(value.columns())
        return columns

    def learned_input_size(self) -> int:
        return sum(value.learned_input_size() for value in self.values)

    def learned_output_size(self) -> int:
        return sum(value.learned_output_size() for value in self.values[:self.num_learned])

    @tensorflow_name_scoped
    def unify_inputs(self, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        x = list()
        index = 0
        for value in self.values:
            num = len(value.learned_input_columns())
            x.append(value.unify_inputs(xs=xs[index: index + num]))
            index += num
        return tf.concat(values=x, axis=-1)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, **kwargs) -> Sequence[tf.Tensor]:
        splits = [value.learned_output_size() for value in self.values[:self.num_learned]]
        ys = tf.split(value=y, num_or_size_splits=splits, axis=1)

        ot = tuple(
            tensor
            for value, y in zip(self.values[:self.num_learned], ys)
            for tensor in value.output_tensors(y=y, **kwargs)
        )[::-1]
        return ot

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        splits = [value.learned_output_size() for value in self.values[:self.num_learned]]
        y = tf.split(value=y, num_or_size_splits=splits, axis=1)
        losses = list()
        index = 0
        for value, y in zip(self.values[:self.num_learned], y):
            num = len(value.learned_output_columns())
            losses.append(value.loss(y=y, xs=xs[index: index + num]))
            index += num
        return tf.math.add_n(inputs=losses)

    @tensorflow_name_scoped
    def distribution_loss(self, samples: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
