from typing import Dict, Sequence, Tuple

import tensorflow as tf

from ..rules import Association


def output_association_tensors(association_rule: Association, y_dict: Dict[str, tf.Tensor]) -> Dict[str, Sequence[tf.Tensor]]:
    """
    Replacement function for value.output_tensors in the case where we want to apply an Association rule. Renormalises
        the joint probabilities of the columns in question with certain outcomes removed by the binding mask in the Association
    Args:
        association_rule: The Association in question to be applied
        y_dict: a full dictionary of outputs from the engine where y_dict[col] gives the appropriate outputs for col

    Returns:
        A dictionary where each key is a column in the association and each value is the sampled output tensor generated
            from the renormalised joint distribution

    Raises:
        ValueError when association rule contains entries that do not exist in the y_dict
    """
    try:
        y_associated = [y_dict[name] for name in association_rule.associations]
        y_associated += [y_dict[f"{name}_nan"] for name in association_rule.nan_associations]
    except KeyError:
        raise ValueError("Association Rule contains column names that are not present in the output of the engine.")

    probs = []
    for y in y_associated:
        y_flat = tf.reshape(y[:, 1:], shape=(-1, y.shape[-1] - 1))
        prob = tf.math.softmax(y_flat, axis=-1)
        probs.append(prob)

    # construct joint distribution and mask out the outputs specified by binding mask
    joint = tf_joint_prob_tensor(*probs)
    masked = tf_masked_probs(joint, association_rule.binding_mask)
    flattened = tf.reshape(masked, (-1, tf.reduce_prod(masked.shape[1:])))

    y = tf.reshape(tf.random.categorical(tf.math.log(flattened), num_samples=1), shape=(-1,))
    output_tensors = unflatten_joint_sample(y, association_rule.binding_mask.shape)

    # each category is shifted along by one to allow the 0th entry to denote NaNs
    for i in range(len(output_tensors)):
        output_tensors[i] = output_tensors[i] + 1

    output_dict: Dict[str, Sequence[tf.Tensor]] = {}
    for i, key in enumerate(association_rule.associations):
        output_dict[key] = (output_tensors[i],)

    for i, key in enumerate(association_rule.nan_associations):
        output_dict[f"{key}_nan"] = (output_tensors[i + len(association_rule.associations)],)

    return output_dict


def tf_joint_prob_tensor(*args: tf.Tensor):
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


def tf_masked_probs(jp: tf.Tensor, mask: tf.Tensor):
    """
    Take joint probability jp and mask outputs that are impossible, renormalise jp now those entries are set to zero
    """
    if jp.shape[1:] != mask.shape:
        raise ValueError("Mask shape doesn't match joint probability's shape (ignoring batch dimension).")

    d = jp * mask
    d = d / tf.reduce_sum(d, axis=range(1, len(jp.shape)), keepdims=True)

    return d


def unflatten_joint_sample(flattened_sample: tf.Tensor, binding_mask_shape: Tuple[int]):
    """
    Reshape sample from a flattened joint probability (bsz, -1), repackage into list where
        output[i] = batch_size number of outputs for column i
    """
    output_tensors = [tf.math.mod(flattened_sample, binding_mask_shape[-1])]
    for n in range(1, len(binding_mask_shape)):
        output_tensors.append(tf.math.mod(
            tf.math.floordiv(
                flattened_sample,
                tf.cast(tf.reduce_prod([binding_mask_shape[-m - 1] for m in range(n)]), dtype=tf.int64)
            ),
            binding_mask_shape[-n - 1]
        ))

    return list(output_tensors[::-1])
