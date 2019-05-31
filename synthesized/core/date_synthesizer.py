from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from .encodings import encoding_modules
from .optimizers import Optimizer
from .synthesizer import Synthesizer
from .transformations import DenseTransformation
from .transformations import transformation_modules
from .values import value_modules


class DateSynthesizer(Synthesizer):

    def __init__(
        self, dtypes, encoding='variational', encoding_size=128, encoder=(64, 64),
        decoder=(64, 64), embedding_size=32, id_embedding_size=128, batch_size=64, iterations=50000
    ):
        super().__init__(name='date_synthesizer')

        self.values = list()
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
            elif dtype.kind == 'f':
                value = self.add_module(
                    module='continuous', modules=value_modules, name=name, positive=True
                )
            elif dtype.kind == 'O':
                value = self.add_module(
                    module='categorical', modules=value_modules, name=name,
                    categories=dtype.categories, embedding_size=embedding_size
                )
            elif dtype.kind == 'M':  # 'm' timedelta
                value = self.add_module(
                    module='date', modules=value_modules, name=name, embedding_size=embedding_size
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
        # dataset = dataset.batch(batch_size=self.batch_size)
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
        identifier, condition = self.identifier_value.random_value(
            n=1, multiples=self.num_synthesize
        )
        x = self.modulation.transform(x=x, condition=condition)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None)
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

    def learn(self, iterations: int=0, data: pd.DataFrame=None, filenames: List[str]=None, verbose: int=0):
        if (data is None) == (filenames is None):
            raise NotImplementedError

        # TODO: increment global step
        if filenames is None:
            assert data is not None
            data = [d[1] for d in data.groupby(by='account_id')]
            for n in range(len(data)):
                for value in self.values:
                    data[n] = value.preprocess(data=data[n])
            data = pd.concat(objs=data)
            num_data = len(data)
            data = {name: d.get_values() for name, d in data.items()}
            fetches = (self.loss, self.optimized)
            for i in range(self.iterations):
                batch = np.random.randint(num_data, size=self.batch_size)
                feed_dict = {name: d[batch] for name, d in data.items()}
                loss, _ = self.run(fetches=fetches, feed_dict=feed_dict)
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
                if verbose:
                    if i % verbose + 100 == verbose:
                        losses = loss / 100.0
                    elif i % verbose + 1 == verbose:
                        losses += loss / 100.0
                        print('{iteration}: {loss:1.2e}'.format(iteration=(i + 1), loss=loss))
                    elif i % verbose + 100 > verbose:
                        print(i % verbose + 100)
                        losses += loss / 100.0

    def synthesize(self, n: int) -> pd.DataFrame:
        synthesized = list()
        for num in n:
            feed_dict = {self.num_synthesize: num}
            x = self.session.run(
                fetches=self.synthesized, feed_dict=feed_dict, options=None, run_metadata=None
            )
            x = pd.DataFrame.from_dict(x)
            for value in self.values:
                x = value.postprocess(data=x)
            synthesized.append(x)
        synthesized = pd.concat(objs=synthesized)
        return synthesized
