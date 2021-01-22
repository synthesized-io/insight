from typing import Any, Dict, List, Tuple

import tensorflow as tf

from ..module import tensorflow_name_scoped
from ..values import Value, DataFrameValue


class Generative(tf.Module):
    """Base class for generative models."""
    def __init__(self, name: str, df_value: DataFrameValue, conditions: List[Value]):
        super(Generative, self).__init__(name=name)

        self.df_value = df_value
        self.conditions = conditions
        self._trainable_variables = None
        self.reconstruction_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.regularization_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.kl_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.total_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)

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

    def specification(self):
        spec = dict(name=self._name)
        return spec

    def get_trainable_variables(self):
        if self._trainable_variables is None:
            self._trainable_variables = self.trainable_variables
        return self._trainable_variables

    def get_variables(self) -> Dict[str, Any]:
        return dict(
            name=self.name
        )

    def set_variables(self, variables: Dict[str, Any]):
        assert variables['name'] == self.name
