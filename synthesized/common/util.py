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


def check_format_version(current_version, old_version):
    try:
        current_version = current_version.split('.', 1)
        old_version = old_version.split('.', 1)

        # Major version
        if current_version[0] != old_version[0]:
            logger.warning("There is a MAJOR incompatibility in the parameters version, "
                           "this may cause unexpected behaviour.")

        # Minor version
        if current_version[1] != old_version[1]:
            logger.warning("There is a minor incompatibility in the parameters version, "
                           "this may cause unexpected behaviour.")

    except Exception as e:
        logger.warning("Couldn't ensure version compatibility. Message: {}".format(e))
