import re
from typing import List, Optional, Dict, Any
from base64 import b64encode, b64decode
import pickle

import tensorflow as tf

from ..module import tensorflow_name_scoped
from ..util import make_tf_compatible


class Value(tf.Module):
    def __init__(self, name: str):
        super().__init__(name=self.__class__.__name__ + '_' + re.sub("\\.", '_', make_tf_compatible(name)))
        self._name = name

        self.placeholder: Optional[tf.Tensor] = None  # not all values have a placeholder
        self.built = False
        self.dtype = tf.float32
        self._regularization_losses: List[tf.Tensor] = list()

    def __str__(self) -> str:
        return self.__class__.__name__[:-5].lower()

    @property
    def name(self):
        return self._name

    def specification(self):
        return dict(name=self._name)

    def columns(self) -> List[str]:
        """External columns which are covered by this value.

        Returns:
            Columns covered by this value.

        """
        return [self.name]

    def learned_input_size(self) -> int:
        """Internal input embedding size for a generative model.

        Returns:
            Learned input embedding size.

        """
        return 0

    def learned_output_size(self) -> int:
        """Internal output embedding size for a generative model.

        Returns:
            Learned output embedding size.

        """
        return 0

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        """Unifies input tensors into a single input embedding for a generative model.

        Args:
            xs: Input tensors, one per `learned_input_columns()`, usually from `input_tensors()`.

        Returns:
            Input embedding, of size `learned_input_size()`.

        """
        raise NotImplementedError

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        """Turns an output embedding of a generative model into corresponding output tensors.

        Args:
            y: Output embedding, of size `learned_output_size()`.

        Returns:
            Output tensors, one per `learned_output_columns()`.

        """
        return list()

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        """Computes the reconstruction loss of an output embedding and corresponding input tensors.

        Args:
            y: Output embedding, of size `learned_output_size()`.
            xs: Input tensors, one per `learned_input_columns()`, usually from `input_tensors()`.

        Returns:
            Loss.

        """
        return tf.constant(value=0.0, dtype=tf.float32)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        """Computes the distributional distance of sample output embeddings to the target
        distribution.

        Args:
            ys: Sample output embeddings, of size `learned_output_size()`.

        Returns:
            Loss.

        """
        return tf.constant(value=0.0, dtype=tf.float32)

    def build(self) -> None:
        self.built = True

    @property
    def regularization_losses(self):
        return self._regularization_losses

    def add_regularization_weight(self, variable: tf.Variable):
        self._regularization_losses.append(variable)
        return variable

    def get_variables(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            pickle=b64encode(pickle.dumps(self)).decode('utf-8')
        )

    @staticmethod
    def set_variables(variables: Dict[str, Any]):
        return pickle.loads(b64decode(variables['pickle'].encode('utf-8')))
