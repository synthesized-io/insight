from typing import List

import faker
import tensorflow as tf

from .value import Value


class BankNumberValue(Value):
    def __init__(self, name):
        super().__init__(name)

    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        return super().unify_inputs(xs=xs)
