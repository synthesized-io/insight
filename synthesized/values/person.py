from typing import List

import tensorflow as tf

from .categorical import CategoricalValue
from .value import Value
from ..common.module import tensorflow_name_scoped


class PersonValue(Value):

    def __init__(self, name, categorical_kwargs: dict, title_label=None, gender_label=None):
        super().__init__(name=name)

        self.title_label = title_label
        self.gender_label = gender_label

        if gender_label is None:
            self.gender = None
        elif self.title_label == self.gender_label:
            # a special case when we extract gender from title:
            self.gender = CategoricalValue(
                name='_gender', **categorical_kwargs
            )
        else:
            self.gender = CategoricalValue(
                name=gender_label, **categorical_kwargs
            )
        self.dtype = tf.int64

    def learned_input_size(self) -> int:
        if self.gender is None:
            return super().learned_input_size()
        else:
            return self.gender.learned_input_size()

    def learned_output_size(self) -> int:
        if self.gender is None:
            return super().learned_output_size()
        else:
            return self.gender.learned_output_size()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        if self.gender is None:
            return super().unify_inputs(xs=xs)
        else:
            return self.gender.unify_inputs(xs=xs)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, sample: bool = False, **kwargs) -> List[tf.Tensor]:
        if self.gender is None:
            return super().output_tensors(y=y, **kwargs)
        else:
            return self.gender.output_tensors(y=y, sample=sample, **kwargs)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        if self.gender is None:
            return super().loss(y=y, xs=xs)
        else:
            return self.gender.loss(y=y, xs=xs)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        if self.gender is None:
            return super().distribution_loss(ys=ys)
        else:
            return self.gender.distribution_loss(ys=ys)
