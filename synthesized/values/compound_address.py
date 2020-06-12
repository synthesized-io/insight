from typing import List

import tensorflow as tf

from .categorical import CategoricalValue
from .value import Value
from ..common.module import tensorflow_name_scoped


class CompoundAddressValue(Value):
    def __init__(self, name, address_label=None):
        super().__init__(name=name)

        self.postcode = CategoricalValue(
            name=address_label, categories=self.postcodes
        )

    def learned_input_columns(self) -> List[str]:
        return self.postcode.learned_input_columns()

    def learned_output_columns(self) -> List[str]:
        return self.postcode.learned_output_columns()

    def learned_input_size(self) -> int:
        return self.postcode.learned_input_size()

    def learned_output_size(self) -> int:
        return self.postcode.learned_output_size()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        return self.postcode.unify_inputs(xs=xs)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        return self.postcode.output_tensors(y=y)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        return self.postcode.loss(y=y, xs=xs)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        return self.postcode.distribution_loss(ys=ys)
