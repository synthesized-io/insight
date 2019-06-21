from typing import Dict, List, Tuple

import tensorflow as tf

from ..module import Module, tensorflow_name_scoped
from ..values import Value


class Generative(Module):
    """Base class for generative models."""

    def __init__(self, name: str, values: List[Value]):
        super().__init__(name=name)

        self.values = values

    @tensorflow_name_scoped
    def learn(self, xs: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Operation]:
        """Training step for the generative model.

        Args:
            xs: An input tensor per value.

        Returns:
            A dictionary of loss tensors, and the optimization operation.

        """
        raise NotImplementedError

    @tensorflow_name_scoped
    def synthesize(self, n: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Generate the given number of instances.

        Args:
            n: The number of instances to generate.

        Returns:
            An output tensor per value.

        """
        raise NotImplementedError
