from typing import Dict, List, Tuple

import tensorflow as tf

from ..module import Module, tensorflow_name_scoped
from ..values import Value


class Generative(Module):

    def __init__(self, name : str, values : List[Value]):
        super().__init__(name=name)

        self.values = values

    @tensorflow_name_scoped
    def learn(self, xs : Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Operation]:
        raise NotImplementedError

    @tensorflow_name_scoped
    def synthesize(self, n : tf.Tensor) -> Dict[str, tf.Tensor]:
        raise NotImplementedError
