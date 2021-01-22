import pickle
import re
from base64 import b64decode, b64encode
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

from ..module import tensorflow_name_scoped
from ..util import make_tf_compatible


class Value(tf.Module):
    def __init__(self, name: str):
        super().__init__(name=self.__class__.__name__ + '_' + re.sub("\\.", '_', make_tf_compatible(name)))
        self._name = name
        self.built = False
        self.dtype = tf.float32
        self._regularization_losses: List[tf.Tensor] = list()

    def __str__(self) -> str:
        return self.__class__.__name__[:-5].lower() + "_value"

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
        return [self._name]

    def split_outputs(self, outputs: Dict[str, Sequence[Any]]) -> Dict[str, np.ndarray]:
        """ Processes output tensors into dict that can be later converted to pandas dataframe

        Args:
            outputs: output tensors from self.output_tensors

        Returns:
            dict of column names to numpy arrays
        """

        try:
            # TODO: need to fix indexing here so that the nasty [0] index can be removed
            outputs = self.convert_tf_to_np_dict({column: outputs[column][0] for column in self.columns()})
        except KeyError:
            raise KeyError(f"Value {self.name} tried to lookup using column name but column wasn't in outputs")
        return outputs

    @staticmethod
    def convert_tf_to_np_dict(tf_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        for name, tensor in tf_dict.items():
            try:
                tf_dict[name] = tensor.numpy()
            except AttributeError:
                tf_dict[name] = tensor

        return tf_dict

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
    def unify_inputs(self, xs: Sequence[tf.Tensor], mask: Optional[tf.Tensor]) -> tf.Tensor:
        """Unifies input tensors into a single input embedding for a generative model.

        Args:
            xs: Input tensors, one per `learned_input_columns()`, usually from `input_tensors()`.
            mask: Optional mask to replace nan inputs
        Returns:
            Input embedding, of size `learned_input_size()`.

        """
        raise NotImplementedError

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, **kwargs) -> Sequence[tf.Tensor]:
        """Turns an output embedding of a generative model into corresponding output tensors.

        Args:
            y: Output embedding, of size `learned_output_size()`.

        Returns:
            Output tensors, one per `learned_output_columns()`.

        """
        return tuple()

    def loss(self, y: tf.Tensor, xs: Sequence[tf.Tensor], mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Computes the reconstruction loss of an output embedding and corresponding input tensors.

        Args:
            y: Output embedding, of size `learned_output_size()`.
            xs: Input tensors, one per `learned_input_columns()`, usually from `input_tensors()`.
            mask: Mask tensor that ignores some inputs from final loss calculation

        Returns:
            Loss.

        """
        return tf.constant(value=0.0, dtype=tf.float32)

    @tensorflow_name_scoped
    def distribution_loss(self, *ys: tf.Tensor) -> tf.Tensor:
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

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'Value':
        return pickle.loads(b64decode(d['pickle'].encode('utf-8')))

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            pickle=b64encode(pickle.dumps(self)).decode('utf-8')
        )
