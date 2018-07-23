import numpy as np
import pandas as pd
import tensorflow as tf

from . import value_modules, transformation_modules, encoding_modules, Synthesizer
from .optimizers import Optimizer
from .transformations import DenseTransformation


class IdSynthesizer(Synthesizer):

    def __init__(
        self, dtypes, encoding='variational', encoding_size=64, encoder=(64, 64), decoder=(64, 64),
        embedding_size=32, id_embedding_size=128, iterations=50000
    ):
        super().__init__(name='id_synthesizer')

        self.values = list()
        self.identifier_value = None
        self.value_output_sizes = list()
        input_size = 0
        output_size = 0
        for name, dtype in zip(dtypes.axes[0], dtypes):
            if name == 'account_id':
                self.identifier_value = self.add_module(
                    module='identifier', modules=value_modules, name=name,
                    num_identifiers=4500, embedding_size=id_embedding_size
                )
                continue
            elif dtype.kind == 'O':
                value = self.add_module(
                    module='categorical', modules=value_modules, name=name,
                    num_categories=len(dtype.categories), embedding_size=embedding_size
                )
            else:
                value = self.add_module(
                    module='continuous', modules=value_modules, name=name, positive=True
                )
            self.values.append(value)
            self.value_output_sizes.append(value.output_size())
            input_size += value.input_size()
            output_size += value.output_size()

        if self.identifier_value is None:
            raise NotImplementedError

        self.encoder = self.add_module(
            module='mlp', modules=transformation_modules, name='encoder',
            input_size=input_size, layer_sizes=encoder
        )

        self.encoding = self.add_module(
            module=encoding, modules=encoding_modules, name='encoding',
            input_size=self.encoder.size(), encoding_size=encoding_size
        )

        self.modulation = self.add_module(
            module='modulation', modules=transformation_modules, name='modulation',
            input_size=encoding_size, condition_size=id_embedding_size
        )

        self.decoder = self.add_module(
            module='mlp', modules=transformation_modules, name='decoder',
            input_size=self.encoding.size(), layer_sizes=decoder
        )

        self.output = self.add_module(
            module=DenseTransformation, name='output', input_size=self.decoder.size(),
            output_size=output_size, batchnorm=False, activation='none'
        )

        # https://twitter.com/karpathy/status/801621764144971776  ;-)
        self.optimizer = self.add_module(
            module=Optimizer, name='optimizer', algorithm='adam', learning_rate=3e-4,
            clip_gradients=1.0
        )

        self.iterations = iterations

    def specification(self):
        spec = super().specification()
        # values?
        spec.update(
            encoding=self.encoding, encoder=self.encoder, decoder=self.decoder,
            iterations=self.iterations
        )
        return spec

    def get_values(self):
        return self.values + [self.identifier_value]

    def tf_initialize(self):
        super().tf_initialize()

        # learn
        xs = list()
        for value in self.values:
            x = value.input_tensor()
            xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.encoding.encode(x=x, encoding_loss=True)
        condition = self.identifier_value.input_tensor()
        x = self.modulation.transform(x=x, condition=condition)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        losses = tf.losses.get_regularization_losses(scope=None)
        for value, x in zip(self.values, xs):
            loss = value.loss(x=x)
            losses.append(loss)
        self.loss = tf.add_n(inputs=losses, name=None)
        # self.loss = tf.losses.get_total_loss(add_regularization_losses=True, name=None)
        self.optimized = self.optimizer.optimize(loss=self.loss)

        # # dataset learn
        # self.filenames = tf.placeholder(dtype=tf.string, shape=(None,), name='filenames')
        # dataset = tf.data.TFRecordDataset(
        #     filenames=self.filenames, compression_type='GZIP', buffer_size=1e7
        #     # num_parallel_reads=None
        # )
        # dataset = dataset.shuffle(buffer_size=10000, seed=None, reshuffle_each_iteration=True)
        # dataset = dataset.map(
        #     map_func=(lambda serialized: tf.parse_single_example(
        #         serialized=serialized,
        #         features={value.name: value.feature() for value in self.get_values()},
        #         name=None, example_names=None
        #     )),
        #     num_parallel_calls=None
        # )
        # dataset = dataset.batch(batch_size=64)
        # self.iterator = dataset.make_initializable_iterator(shared_name=None)
        # next_values = self.iterator.get_next()
        # xs = list()
        # for value in self.values:
        #     x = value.input_tensor(feed=next_values[value.name])
        #     xs.append(x)
        # x = tf.concat(values=xs, axis=1, name=None)
        # x = self.encoder.transform(x=x)
        # x = self.encoding.encode(x=x, encoding_loss=True)
        # condition = self.identifier_value.input_tensor(
        #     feed=next_values[self.identifier_value.name]
        # )
        # x = self.modulation.transform(x=x, condition=condition)
        # x = self.decoder.transform(x=x)
        # x = self.output.transform(x=x)
        # xs = tf.split(
        #     value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        # )
        # losses = tf.losses.get_regularization_losses(scope=None)
        # for value, x in zip(self.values, xs):
        #     loss = value.loss(x=x, feed=next_values[value.name])
        #     losses.append(loss)
        # self.loss_dataset = tf.add_n(inputs=losses, name=None)
        # # self.loss_dataset = tf.losses.get_total_loss(add_regularization_losses=True, name=None)
        # self.optimized_dataset = self.optimizer.optimize(loss=self.loss_dataset)

        # synthesize
        self.num_synthesize = tf.placeholder(dtype=tf.int64, shape=(), name='num-synthesize')
        x = self.encoding.sample(n=self.num_synthesize)
        identifier, condition = self.identifier_value.random_value(n=self.num_synthesize)
        x = self.modulation.transform(x=x, condition=condition)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None)
        self.synthesized = dict()
        for value, x in zip(self.values, xs):
            self.synthesized[value.name] = value.output_tensor(x=x)
        self.synthesized[self.identifier_value.name] = identifier

        # transform
        xs = list()
        for value in self.values:
            x = value.input_tensor()
            xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.encoding.encode(x=x)
        condition = self.identifier_value.input_tensor()
        x = self.modulation.transform(x=x, condition=condition)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        self.transformed = dict()
        for value, x in zip(self.values, xs):
            self.transformed[value.name] = value.output_tensor(x=x)
        self.transformed[self.identifier_value.name] = self.identifier_value.placeholder

    def learn(self, data=None, filenames=None, verbose=0):
        if (data is None) == (filenames is None):
            raise NotImplementedError

        # TODO: increment global step
        if filenames is None:
            data = [data[value.name].get_values() for value in self.get_values()]
            num_data = len(data[0])
            fetches = (self.loss, self.optimized)
            for i in range(self.iterations):
                batch = np.random.randint(num_data, size=64)
                feed_dict = {
                    value.placeholder: d[batch] for value, d in zip(self.get_values(), data)
                }
                loss, _ = self.session.run(
                    fetches=fetches, feed_dict=feed_dict, options=None, run_metadata=None
                )
                if verbose and i % verbose + 1 == verbose:
                    print('{iteration}: {loss:1.2e}'.format(iteration=(i + 1), loss=loss))

        else:
            fetches = self.iterator.initializer
            feed_dict = {self.filenames: filenames}
            self.session.run(fetches=fetches, feed_dict=feed_dict, options=None, run_metadata=None)
            fetches = (self.loss_dataset, self.optimized_dataset)
            # TODO: while loop for training
            for i in range(self.iterations):
                loss, _ = self.session.run(
                    fetches=fetches, feed_dict=None, options=None, run_metadata=None
                )
                if verbose and i % verbose + 1 == verbose:
                    print('{iteration}: {loss:1.2e}'.format(iteration=(i + 1), loss=loss))

    def synthesize(self, n):
        feed_dict = {self.num_synthesize: n}
        synthesized = self.session.run(
            fetches=self.synthesized, feed_dict=feed_dict, options=None, run_metadata=None
        )
        return pd.DataFrame.from_dict(synthesized)

    def transform(self, X, **transform_params):
        assert not transform_params
        feed_dict = {value.placeholder: X[value.name].get_values() for value in self.get_values()}
        transformed = self.session.run(
            fetches=self.transformed, feed_dict=feed_dict, options=None, run_metadata=None
        )
        return pd.DataFrame.from_dict(transformed)
