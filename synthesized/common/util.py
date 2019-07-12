import tensorflow as tf


def get_initializer(initializer):
    if initializer == 'normal':
        return tf.random_normal_initializer(mean=0.0, stddev=1e-2)
    elif initializer == 'normal-large':
        return tf.random_normal_initializer(mean=0.0, stddev=1.0)
    elif initializer == 'orthogonal':
        return tf.orthogonal_initializer(gain=1.0)
    elif initializer == 'ones':
        return tf.ones_initializer(dtype=tf.float32)
    elif initializer == 'zeros':
        return tf.zeros_initializer(dtype=tf.float32)
    elif initializer == 'zeros-int':
        return tf.zeros_initializer(dtype=tf.int64)
    else:
        raise NotImplementedError


def get_regularizer(regularizer, weight):
    assert weight >= 0.0
    if regularizer == 'none' or weight == 0.0:
        return tf.no_regularizer
    elif regularizer == 'l2':
        return tf.contrib.layers.l2_regularizer(scale=weight, scope=None)
    else:
        raise NotImplementedError
