import re
from typing import List
import logging

import tensorflow as tf

from .categorical import CategoricalValue
from .value import Value
from ..common.module import tensorflow_name_scoped

logger = logging.getLogger(__name__)


class AddressValue(Value):
    postcode_regex = re.compile(r'^[A-Za-z]{1,2}[0-9]+[A-Za-z]? *[0-9]+[A-Za-z]{2}$')

    def __init__(self, name, categorical_kwargs: dict, postcode_label: str = None,
                 fake: bool = True):

        super().__init__(name=name)

        self.postcode_label = postcode_label

        if fake:
            self.fake = True
            self.postcode = None
        else:
            self.fake = False
            self.postcode = CategoricalValue(name=postcode_label, **categorical_kwargs)

        self.dtype = tf.int64
        assert self.fake or self.postcode

    def learned_input_columns(self) -> List[str]:
        if self.postcode is None:
            return super().learned_input_columns()
        else:
            return self.postcode.learned_input_columns()

    def learned_output_columns(self) -> List[str]:
        if self.postcode is None:
            return super().learned_output_columns()
        else:
            return self.postcode.learned_output_columns()

    def learned_input_size(self) -> int:
        if self.postcode is None:
            return super().learned_input_size()
        else:
            return self.postcode.learned_input_size()

    def learned_output_size(self) -> int:
        if self.postcode is None:
            return super().learned_output_size()
        else:
            return self.postcode.learned_output_size()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        if self.postcode is None:
            return super().unify_inputs(xs=xs)
        else:
            return self.postcode.unify_inputs(xs=xs)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, sample: bool = False, **kwargs) -> List[tf.Tensor]:
        if self.postcode is None:
            return super().output_tensors(y=y, **kwargs)
        else:
            return self.postcode.output_tensors(y=y, sample=sample, **kwargs)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        if self.postcode is None:
            return super().loss(y=y, xs=xs)
        else:
            return self.postcode.loss(y=y, xs=xs)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        if self.postcode is None:
            return super().distribution_loss(ys=ys)
        else:
            return self.postcode.distribution_loss(ys=ys)
