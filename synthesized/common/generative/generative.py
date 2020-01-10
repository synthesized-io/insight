from typing import Dict, List, Tuple

import tensorflow as tf

from ..module import Module, tensorflow_name_scoped
from ..values import Value


class Generative(Module):
    """Base class for generative models."""

    def __init__(self, name: str, values: List[Value], conditions: List[Value]):
        super().__init__(name=name)

        self.values = values
        self.conditions = conditions

    def module_initialize(self):
        super().module_initialize()
        self.global_step = tf.compat.v1.train.get_global_step()

    @tensorflow_name_scoped
    def learn(self, xs: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Operation]:
        """Training step for the generative model.

        Args:
            xs: Input tensor per column.

        Returns:
            Dictionary of loss tensors, and optimization operation.

        """
        raise NotImplementedError

    @tensorflow_name_scoped
    def synthesize(self, n: tf.Tensor, cs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Generate the given number of instances.

        Args:
            n: Number of instances to generate.
            cs: Condition tensor per column.

        Returns:
            Output tensor per column.

        """
        raise NotImplementedError
