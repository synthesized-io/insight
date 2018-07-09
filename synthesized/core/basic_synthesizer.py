import numpy as np
import pandas as pd
import tensorflow as tf
from synthesized.core import value_modules, transformation_modules, encoding_modules, Synthesizer


class BasicSynthesizer(Synthesizer):

    def __init__(
        self, dtypes, encoding='variational', encoding_size=64, encoder=(64,), decoder=(64,)
    ):
        super().__init__(name='basic_synthesizer')
        self.values = list()
        self.value_sizes = list()
        self.data_size = 0
        submodules = list()
        for name, dtype in zip(dtypes.axes[0], dtypes):
            if dtype.kind == 'O':
                value = self.add_module(
                    module='categorical', modules=value_modules, name=name,
                    num_categories=len(dtype.categories)
                )
            else:
                value = self.add_module(
                    module='continuous', modules=value_modules, name=name, positive=False
                )
            self.values.append(value)
            self.value_sizes.append(value.size())
            self.data_size += value.size()
            submodules.append(value)

        self.encoder = self.add_module(
            module='mlp', modules=transformation_modules, name='encoder',
            input_size=self.data_size, output_size=encoding_size, layer_sizes=encoder
        )
        submodules.append(self.encoder)

        self.encoding = self.add_module(module=encoding, modules=encoding_modules, name='encoding', encoding_size=encoding_size)
        submodules.append(self.encoding)

        self.decoder = self.add_module(
            module='mlp', modules=transformation_modules, name='decoder',
            input_size=encoding_size, output_size=self.data_size, layer_sizes=decoder
        )
        submodules.append(self.decoder)

    def _initialize(self):
        # learn
        xs = list()
        for value in self.values:
            xs.append(value.input_tensor())
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.encoding.encode(x=x, encoding_loss=True)
        x = self.decoder.transform(x=x)
        xs = tf.split(value=x, num_or_size_splits=self.value_sizes, axis=1, num=None, name=None)
        for value, x in zip(self.values, xs):
            value.loss(x=x)

        self.loss = tf.losses.get_total_loss(add_regularization_losses=True, name=None)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=3e-4,  # https://twitter.com/karpathy/status/801621764144971776  ;-)
            beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name='Adam'
        )
        grads_and_vars = self.optimizer.compute_gradients(
            loss=self.loss, var_list=None, aggregation_method=None,
            colocate_gradients_with_ops=False, grad_loss=None  # gate_gradients=GATE_OP
        )
        grads_and_vars = [
            (tf.clip_by_value(t=grad, clip_value_min=-1.0, clip_value_max=1.0, name=None), var)
            for grad, var in grads_and_vars
        ]
        self.optimized = self.optimizer.apply_gradients(
            grads_and_vars=grads_and_vars, global_step=None, name=None
        )
        # self.optimized = self.optimizer.minimize(
        #     loss=self.loss, global_step=None, var_list=None, aggregation_method=None,
        #     colocate_gradients_with_ops=False, name=None, grad_loss=None  # gate_gradients=GATE_OP
        # )

        # synthesize
        self.num_synthesize = tf.placeholder(dtype=tf.int64, shape=(), name='num-synthesize')
        x = self.encoding.sample(n=self.num_synthesize)
        x = self.decoder.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_sizes, axis=1, num=None, name=None
        )
        self.synthesized = dict()
        for value, x in zip(self.values, xs):
            self.synthesized[value.name] = value.output_tensor(x=x)

        # transform
        xs = list()
        for value in self.values:
            xs.append(value.input_tensor())
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.encoding.encode(x=x)
        x = self.decoder.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_sizes, axis=1, num=None, name=None
        )
        self.transformed = dict()
        for value, x in zip(self.values, xs):
            self.transformed[value.name] = value.output_tensor(x=x)

    def learn(self, data, verbose=0):
        data = [data[value.name].get_values() for value in self.values]
        num_data = len(data[0])
        for i in range(50000):
            batch = np.random.randint(num_data, size=64)
            feed_dict = {value.placeholder: d[batch] for value, d in zip(self.values, data)}
            loss, _ = self.session.run(
                fetches=(self.loss, self.optimized), feed_dict=feed_dict, options=None,
                run_metadata=None
            )
            if verbose and i % verbose + 1 == verbose:
                print('{iteration}: {loss:1.2e}'.format(iteration=(i + 1), loss=loss))
            #     print('{iteration}: {loss:1.2e} ({details})'.format(
            #         iteration=(i + 1), loss=loss, details=', '.join(
            #             '{name}={loss:1.2e}'.format(name=name, loss=loss)
            #             for name, loss in losses.items()
            #         )
            #     ))

    def synthesize(self, n):
        feed_dict = {self.num_synthesize: n}
        synthesized = self.session.run(
            fetches=self.synthesized, feed_dict=feed_dict, options=None, run_metadata=None
        )
        return pd.DataFrame.from_dict(synthesized)

    def transform(self, X, **transform_params):
        assert not transform_params
        feed_dict = {value.placeholder: X[value.name].get_values() for value in self.values}
        transformed = self.session.run(
            fetches=self.transformed, feed_dict=feed_dict, options=None, run_metadata=None
        )
        return pd.DataFrame.from_dict(transformed)
