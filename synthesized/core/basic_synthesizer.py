import numpy as np
import pandas as pd
import tensorflow as tf
from synthesized.core import Synthesizer


class BasicSynthesizer(Synthesizer):

    def __init__(self, dtypes, encoded_size):
        super().__init__(dtypes=dtypes)
        self.inputs = dict()
        self.losses = dict()
        self.loss = None
        self.optimized = None
        self.num_synthesize = None
        self.session = None

        input_sizes = list()
        for dtype in self.dtypes:
            if dtype.kind == 'O':
                input_sizes.append(len(dtype.categories))
            else:
                input_sizes.append(1)
        input_names = list(self.dtypes.axes[0])
        data_size = sum(input_sizes)

        # weights
        encoder_weights1 = tf.get_variable(
            name='encoder_weights1', shape=(data_size, encoded_size), dtype=tf.float32,
            initializer=None, regularizer=None, trainable=True, collections=None,
            caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
            custom_getter=None, constraint=None
        )
        encoder_weights2 = tf.get_variable(
            name='encoder_weights2', shape=(encoded_size, encoded_size), dtype=tf.float32,
            initializer=None, regularizer=None, trainable=True, collections=None,
            caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
            custom_getter=None, constraint=None
        )
        decoder_weights1 = tf.get_variable(
            name='decoder_weights1', shape=(encoded_size, encoded_size), dtype=tf.float32,
            initializer=None, regularizer=None, trainable=True, collections=None,
            caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
            custom_getter=None, constraint=None
        )
        decoder_weights2 = tf.get_variable(
            name='decoder_weights2', shape=(encoded_size, data_size), dtype=tf.float32,
            initializer=None, regularizer=None, trainable=True, collections=None,
            caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
            custom_getter=None, constraint=None
        )

        # fit
        original = list()
        for name, dtype in zip(input_names, self.dtypes):
            if dtype.kind == 'O':
                num_categories = len(dtype.categories)
                self.inputs[name] = tf.placeholder(
                    dtype=tf.int64, shape=(None,), name=(name + '-input')
                )
                original.append(tf.one_hot(
                    indices=self.inputs[name], depth=num_categories, on_value=1.0, off_value=0.0,
                    axis=1, dtype=tf.float32, name=None
                ))
            else:
                self.inputs[name] = tf.placeholder(
                    dtype=tf.float32, shape=(None,), name=(name + '-input')
                )
                original.append(tf.expand_dims(input=self.inputs[name], axis=1, name=None))
        encoded = tf.concat(values=original, axis=1, name=None)
        encoded = tf.matmul(a=encoded, b=encoder_weights1, name=None)
        encoded = tf.nn.relu(features=encoded, name=None)
        encoded = tf.matmul(a=encoded, b=encoder_weights2, name=None)
        encoded = tf.tanh(x=encoded, name=None)
        decoded = tf.matmul(a=encoded, b=decoder_weights1, name=None)
        decoded = tf.nn.relu(features=decoded, name=None)
        decoded = tf.matmul(a=decoded, b=decoder_weights2, name=None)
        decoded = tf.split(
            value=decoded, num_or_size_splits=input_sizes, axis=1, num=None, name=None
        )
        for name, dtype, target, tensor in zip(input_names, self.dtypes, original, decoded):
            if dtype.kind == 'O':
                num_categories = len(dtype.categories)
                self.losses[name] = tf.losses.softmax_cross_entropy(
                    onehot_labels=target, logits=tensor, weights=1.0, label_smoothing=0,
                    scope=None, loss_collection=tf.GraphKeys.LOSSES
                )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
            else:
                tensor = tf.exp(x=tensor, name=None)
                self.losses[name] = tf.losses.mean_squared_error(
                    labels=target, predictions=tensor, weights=1.0, scope=None,
                    loss_collection=tf.GraphKeys.LOSSES
                )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        self.loss = tf.losses.get_total_loss(add_regularization_losses=True, name=None)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=3e-4,  # https://twitter.com/karpathy/status/801621764144971776  ;-)
            beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam'
        )
        self.optimized = optimizer.minimize(
            loss=self.loss, global_step=None, var_list=None, aggregation_method=None,
            colocate_gradients_with_ops=False, name=None, grad_loss=None  # gate_gradients=GATE_OP
        )

        # synthesize
        self.num_synthesize = tf.placeholder(dtype=tf.int32, shape=(), name=None)
        encoded = tf.random_uniform(
            shape=(self.num_synthesize, encoded_size), minval=-1.0, maxval=1.0, dtype=tf.float32,
            seed=None, name=None
        )
        # tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
        decoded = tf.matmul(a=encoded, b=decoder_weights1, name=None)
        decoded = tf.nn.relu(features=decoded, name=None)
        decoded = tf.matmul(a=decoded, b=decoder_weights2, name=None)
        decoded = tf.split(
            value=decoded, num_or_size_splits=input_sizes, axis=1, num=None, name=None
        )
        self.synthesized = dict()
        for name, dtype, tensor in zip(input_names, self.dtypes, decoded):
            if dtype.kind == 'O':
                self.synthesized[name] = tf.argmax(input=tensor, axis=1, name=None)
            else:
                tensor = tf.exp(x=tensor, name=None)
                self.synthesized[name] = tf.squeeze(input=tensor, axis=1, name=None)

        # tensorflow
        initialize = tf.global_variables_initializer()
        self.session = tf.Session(target='', graph=None, config=None)
        self.session.__enter__()
        self.session.run(fetches=initialize, feed_dict=None, options=None, run_metadata=None)

    def fit(self, data):
        print('=== optimization ===')
        data = {name: data[name].get_values() for name in self.dtypes.axes[0]}
        num_data = len(data[self.dtypes.axes[0][0]])
        for i in range(50000):
            batch = np.random.randint(num_data, size=64)
            feed_dict = {self.inputs[name]: values[batch] for name, values in data.items()}
            loss, losses, _ = self.session.run(
                fetches=(self.loss, self.losses, self.optimized), feed_dict=feed_dict,
                options=None, run_metadata=None
            )
            if i % 1000 == 999:
                print('{iteration}: {loss:1.2e} ({details})'.format(
                    iteration=(i + 1), loss=loss, details=', '.join(
                        '{name}={loss:1.2e}'.format(name=name, loss=loss)
                        for name, loss in losses.items()
                    )
                ))
        print('===== finished =====')

    def synthesize(self, n):
        feed_dict = {self.num_synthesize: n}
        synthesized = self.session.run(
            fetches=self.synthesized, feed_dict=feed_dict, options=None, run_metadata=None
        )
        return pd.DataFrame.from_dict(synthesized)
