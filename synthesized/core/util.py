import tensorflow as tf


initializers = dict(
    normal=tf.random_normal_initializer(mean=0.0, stddev=1e-2, seed=None, dtype=tf.float32)
)

regularizers = dict(
    l2=tf.contrib.layers.l2_regularizer(scale=1e-5, scope=None)
)
