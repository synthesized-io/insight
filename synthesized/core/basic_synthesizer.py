import numpy as np
import pandas as pd
import tensorflow as tf

from .encodings import encoding_modules
from .optimizers import Optimizer
from .synthesizer import Synthesizer
from .transformations import DenseTransformation
from .transformations import transformation_modules
from .values import value_modules


class BasicSynthesizer(Synthesizer):

    def __init__(
        self, dtypes, encoding='variational', encoding_size=64, encoder=(64, 64), decoder=(64, 64),
        embedding_size=32, batch_size=64, iterations=50000
    ):
        super().__init__(name='basic_synthesizer')

        self.values = list()
        self.value_output_sizes = list()
        input_size = 0
        output_size = 0
        for name, dtype in zip(dtypes.axes[0], dtypes):
            if dtype.kind == 'f':
                value = self.add_module(
                    module='continuous', modules=value_modules, name=name, positive=True
                )
            elif dtype.kind == 'O':
                value = self.add_module(
                    module='categorical', modules=value_modules, name=name,
                    categories=dtype.categories, embedding_size=embedding_size
                )
            else:
                raise NotImplementedError
            self.values.append(value)
            self.value_output_sizes.append(value.output_size())
            input_size += value.input_size()
            output_size += value.output_size()

        self.encoder = self.add_module(
            module='mlp', modules=transformation_modules, name='encoder',
            input_size=input_size, layer_sizes=encoder
        )

        self.encoding = self.add_module(
            module=encoding, modules=encoding_modules, name='encoding',
            input_size=self.encoder.size(), encoding_size=encoding_size
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

        self.batch_size = batch_size
        self.iterations = iterations

    def specification(self):
        spec = super().specification()
        # values?
        spec.update(
            encoding=self.encoding, encoder=self.encoder, decoder=self.decoder,
            batch_size=self.batch_size, iterations=self.iterations
        )
        return spec

    def get_values(self):
        return list(self.values)

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
        #     filenames=self.filenames, compression_type='GZIP', buffer_size=1e6
        #     # num_parallel_reads=None
        # )
        # dataset = dataset.shuffle(buffer_size=10000, seed=None, reshuffle_each_iteration=True)
        # dataset = dataset.map(
        #     map_func=(lambda serialized: tf.parse_single_example(
        #         serialized=serialized,
        #         features={value.name: value.feature() for value in self.values},
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
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None)
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
        x = self.output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        self.transformed = dict()
        for value, x in zip(self.values, xs):
            self.transformed[value.name] = value.output_tensor(x=x)

    def learn(self, data=None, filenames=None, verbose=0):
        if (data is None) == (filenames is None):
            raise NotImplementedError

        # TODO: increment global step
        if filenames is None:
            for value in self.values:
                data = value.preprocess(data=data)
            data = [data[value.name].get_values() for value in self.get_values()]
            num_data = len(data[0])
            fetches = (self.loss, self.optimized)
            for i in range(self.iterations):
                batch = np.random.randint(num_data, size=self.batch_size)
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
        synthesized = pd.DataFrame.from_dict(synthesized)
        for value in self.values:
            synthesized = value.postprocess(data=synthesized)
        return synthesized

    def transform(self, X, **transform_params):
        assert not transform_params
        for value in self.values:
            X = value.preprocess(data=X)
        feed_dict = {value.placeholder: X[value.name].get_values() for value in self.get_values()}
        transformed = self.session.run(
            fetches=self.transformed, feed_dict=feed_dict, options=None, run_metadata=None
        )
        transformed = pd.DataFrame.from_dict(transformed)
        for value in self.values:
            transformed = value.postprocess(data=transformed)
        return transformed
