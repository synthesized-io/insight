import io
import logging
import re

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.eager import context

logger = logging.getLogger(__name__)

RE_START = re.compile(r"^[^A-Za-z0-9.]")
RE_END = re.compile(r"[^A-Za-z0-9.]")


def record_summaries_every_n_global_steps(n: int, global_step: tf.Variable):
    """Sets the should_record_summaries Tensor to true if global_step % n == 0."""
    with tf.device("cpu:0"):
        if n != 0:
            def should():
                return tf.math.equal(global_step % tf.constant(n, dtype=tf.int64), 0)

            if not context.executing_eagerly():
                should = should()
        else:
            should = tf.constant(False, dtype=tf.bool)

    return tf.summary.record_if(should)


def plot_to_tf_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def get_initializer(initializer):
    if initializer == 'normal':
        return tf.initializers.RandomNormal(mean=0.0, stddev=1e-2)
    elif initializer == 'normal-small':
        return tf.initializers.RandomNormal(mean=0.0, stddev=1e-3)
    elif initializer == 'normal-large':
        return tf.initializers.RandomNormal(mean=0.0, stddev=1.0)
    elif initializer == 'glorot-normal':
        return tf.initializers.glorot_normal()
    elif initializer == 'orthogonal':
        return tf.initializers.orthogonal(gain=1.0)
    elif initializer == 'orthogonal-small':
        return tf.initializers.orthogonal(gain=1e-2)
    elif initializer == 'ones':
        return tf.initializers.ones()
    elif initializer == 'zeros':
        return tf.initializers.zeros()

    else:
        raise NotImplementedError


def get_regularizer(regularizer, weight):
    assert weight >= 0.0
    if regularizer == 'none' or weight == 0.0:
        return tf.compat.v1.no_regularizer
    elif regularizer == 'l2':
        return tf.keras.regularizers.l2(0.5 * (weight))
    else:
        raise NotImplementedError


def make_tf_compatible(string):
    return re.sub(RE_END, '_', re.sub(RE_START, '.', str(string)))


class IncompatibleVersionException(Exception):
    """Base exception class for incompatible versions."""


def check_version(current_version, old_version, major_compatibility=None):
    """Check if current version is compatible with given (old) version, checking major compatibility. If they are not
    compatible, raise IncompatibleVersionException.

    Args:
        current_version:
        old_version:
        major_compatibility: List of major versions that current version is compatible with.

    Example:
        check_version('3.1', '3.0') -> Won't rise exception.
        check_version('3.1', '1.0', major_compatibility=['1', '2']) -> Won't rise exception.
        check_version('3.1', '1.0', major_compatibility=['2']) -> WILL rise exception.

    """
    major_compatibility = major_compatibility if major_compatibility is not None else []

    # Minor version
    if current_version != old_version:
        logger.debug(f"Given and current versions are different ({old_version}, {current_version}), "
                     f"this may cause unexpected behaviour.")

    current_version = current_version.split('.', 1)[0]
    old_version = old_version.split('.', 1)[0]

    # Major version
    if current_version != old_version and current_version not in major_compatibility:
        raise IncompatibleVersionException(f"Given version {old_version} for the imported model is not compatible with "
                                           f"current version {current_version}.")
