import logging
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from synthesized.common.module import tensorflow_name_scoped

from .categorical import CategoricalValue
from .value import Value

logger = logging.getLogger(__name__)


class AssociatedCategoricalValue(Value, Mapping):
    def __init__(
            self, values: Dict[str, CategoricalValue], binding_mask: np.ndarray, name: Optional[str] = None,
    ):
        if name is None:
            name = '|'.join(values)
        super(AssociatedCategoricalValue, self).__init__(
            name=name,
        )
        self.dtype = tf.int64
        self._values: Dict[str, Value] = OrderedDict(**values)
        self.binding_mask: tf.Tensor = tf.constant(binding_mask, dtype=tf.float32)

    def __str__(self) -> str:
        string = super().__str__()
        return string

    def __getitem__(self, k: str) -> Value:
        return self._values[k]

    def __iter__(self) -> Iterator[str]:
        for key in self._values:
            yield key

    def __len__(self) -> int:
        return len(self._values)

    def columns(self) -> List[str]:
        return [column for value in self.values() for column in value.columns()]

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            categories=self.categories, embedding_size=self.embedding_size,
            similarity_based=self.similarity_based,
            weight=self.weight, temperature=self.temperature, moving_average=self.use_moving_average,
            embedding_initialization=self.embedding_initialization
        )
        return spec

    def learned_input_size(self) -> int:
        return sum([v.learned_input_size() for v in self.values()])

    def learned_output_size(self) -> int:
        return sum([v.learned_output_size() for v in self.values()])

    @tensorflow_name_scoped
    def unify_inputs(self, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        inputs = [value.unify_inputs(xs[n:n + 1]) for n, value in enumerate(self.values())]
        return tf.concat(inputs, axis=-1)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, **kwargs) -> Sequence[tf.Tensor]:
        """Outputs the bound categorical values."""
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values()],
            axis=-1
        )

        probs = []
        for y in ys:
            y_flat = tf.reshape(y[:, 1:], shape=(-1, y.shape[-1] - 1))
            prob = tf.math.softmax(y_flat, axis=-1)
            probs.append(prob)

        # construct joint distribution and mask out the outputs specified by binding mask
        joint = tf_joint_prob_tensor(*probs)
        masked = tf_masked_probs(joint, self.binding_mask)
        flattened = tf.reshape(masked, (-1, tf.reduce_prod(masked.shape[1:])))

        y = tf.reshape(tf.random.categorical(tf.math.log(flattened), num_samples=1), shape=(-1,))
        output_tensors = unflatten_joint_sample(y, list(self.values()))
        for i in range(len(output_tensors)):
            output_tensors[i] = output_tensors[i] + 1

        return output_tensors

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values()],
            axis=-1
        )
        return tf.reduce_sum([v.loss(y=ys[n], xs=xs[n:n + 1]) for n, v in enumerate(self.values())], axis=None)

    def split_inputs(self, xs: Sequence[tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Concatenate input tensors per value
        return {name: xs[i] for i, name in enumerate(self)}

    def split_outputs(self, outputs: Dict[str, Sequence[Any]]) -> Dict[str, np.ndarray]:
        output_dict = {}
        assert len(outputs) == 1
        output_tup: Sequence[Any] = next(iter(outputs.values()))
        for (name, value), x in zip(self.items(), output_tup):
            output_dict.update(self.convert_tf_to_np_dict(value.split_outputs({name: (x,)})))
        return output_dict


def tf_joint_prob_tensor(*args):
    """
    Constructs a tensor where each element represents a single joint probability, when len(args) > 4 this can
        become computationally costly, especially when each argument has many categories

    Args
        *args: container where args[i] is the tensor of probabilities with batch dimension at 0th position

    Returns
        Tensor where element [b, i, j, k] equals joint probability bth batch has
            i in 1st dim, j in 2nd dim and k in 3rd dim
    """
    rank = len(args)
    probs = []

    for n, x in enumerate(args):
        for m in range(n):
            x = tf.expand_dims(x, axis=-2 - m)
        for m in range(rank - n - 1):
            x = tf.expand_dims(x, axis=-1)
        probs.append(x)

    joint_prob = probs[0]
    for n in range(1, rank):
        joint_prob = joint_prob * probs[n]

    return joint_prob


def tf_masked_probs(jp, mask):
    """
    Take joint probability jp and mask outputs that are impossible, renormalise jp now those entries are set to zero
    """
    if jp.shape[1:] != mask.shape:
        raise ValueError("Mask shape doesn't match joint probability's shape (ignoring batch dimension).")

    d = jp * mask
    d = d / tf.reduce_sum(d, axis=range(1, len(jp.shape)), keepdims=True)

    return d


def unflatten_joint_sample(flattened_sample, value_list):
    """
    Reshape sample from a flattened joint probability (bsz, -1), repackage into output tensor
        size (bsz, self.value[0].num_categories, ...)
    """
    output_tensors = [tf.math.mod(flattened_sample, value_list[-1].num_categories)]
    for n in range(1, len(value_list)):
        output_tensors.append(tf.math.mod(
            tf.math.floordiv(
                flattened_sample,
                tf.cast(tf.reduce_prod([value_list[-m - 1].num_categories for m in range(n)]), dtype=tf.int64)
            ),
            value_list[-n - 1].num_categories
        ))

    return list(output_tensors[::-1])
