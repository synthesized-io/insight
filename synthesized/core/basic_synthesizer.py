import numpy as np
import pandas as pd
import tensorflow as tf

from .encodings import encoding_modules
from .module import Module
from .optimizers import Optimizer
from .synthesizer import Synthesizer
from .transformations import transformation_modules
from .values import identify_value


class BasicSynthesizer(Synthesizer):

    def __init__(
        self, data, exclude_encoding_loss=False,
        # architecture
        network_type='resnet', encoding='variational',
        # hyperparameters
        capacity=64, depth=4, learning_rate=3e-4, batch_size=64,
        # person
        gender_label=None, name_label=None, firstname_label=None, lastname_label=None,
        email_label=None,
        # address
        postcode_label=None, street_label=None,
        # identifier
        identifier_label=None
    ):
        super().__init__(name='synthesizer')

        self.exclude_encoding_loss = exclude_encoding_loss

        self.capacity = capacity
        self.depth = depth
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # person
        self.person_value = None
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        # address
        self.address_value = None
        self.postcode_label = postcode_label
        self.street_label = street_label
        # identifier
        self.identifier_value = None
        self.identifier_label = identifier_label
        # date
        self.date_value = None

        self.values = list()
        self.value_output_sizes = list()
        input_size = 0
        output_size = 0
        print('value types:')
        for name, dtype in zip(data.dtypes.axes[0], data.dtypes):
            value = self.get_value(name=name, dtype=dtype, data=data)
            print(name, value)
            if value is not None:
                value.extract(data=data)
                self.values.append(value)
                self.value_output_sizes.append(value.output_size())
                input_size += value.input_size()
                output_size += value.output_size()

        self.encoder = self.add_module(
            module=network_type, modules=transformation_modules, name='encoder',
            input_size=input_size, layer_sizes=[self.capacity for _ in range(self.depth)]
        )

        self.encoding = self.add_module(
            module=encoding, modules=encoding_modules, name='encoding',
            input_size=self.encoder.size(), encoding_size=self.capacity
        )

        self.decoder = self.add_module(
            module=network_type, modules=transformation_modules, name='decoder',
            input_size=self.encoding.size(), layer_sizes=[self.capacity for _ in range(self.depth)]
        )

        self.output = self.add_module(
            module='dense', modules=transformation_modules, name='output',
            input_size=self.decoder.size(), output_size=output_size, batchnorm=False,
            activation='none', regularizer='none'
        )

        # https://twitter.com/karpathy/status/801621764144971776  ;-)
        self.optimizer = self.add_module(
            module=Optimizer, name='optimizer', algorithm='adam', learning_rate=self.learning_rate,
            clip_gradients=1.0
        )

    def specification(self):
        spec = super().specification()
        # TODO: values?
        spec.update(
            encoding=self.encoding, encoder=self.encoder, decoder=self.decoder,
            batch_size=self.batch_size
        )
        return spec

    def get_value(self, name, dtype, data):
        return identify_value(module=self, name=name, dtype=dtype, data=data)

    # not needed?
    def preprocess(self, data):
        for value in self.values:
            data = value.preprocess(data=data)
        return data

    def customized_transform(self, x):
        return x

    def customized_synthesize(self, x):
        return x

    def tf_train_iteration(self, feed=None):
        if feed is None:
            feed = dict()
        xs = list()
        for value in self.values:
            for label in value.trainable_labels():
                # critically assumes max one trainable label
                x = value.input_tensor(feed=feed.get(value.name))
                xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x, encoding_loss = self.encoding.encode(x=x, encoding_loss=True)
        x = self.customized_transform(x=x)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        reg_loss = tf.add_n(inputs=tf.losses.get_regularization_losses(scope=None))
        losses = dict(encoding=encoding_loss, regularization=reg_loss)
        if self.exclude_encoding_loss:
            losses.pop('encoding')
        for value, x in zip(self.values, xs):
            for label in value.trainable_labels():
                loss = value.loss(x=x, feed=feed.get(value.name))
                if loss is not None:
                    losses[label] = loss
        loss = tf.add_n(inputs=[losses[name] for name in sorted(losses)])
        optimized = self.optimizer.optimize(loss=loss)
        return losses, loss, optimized

    def tf_initialize(self):
        super().tf_initialize()

        # learn
        self.losses, self.loss, self.optimized = self.train_iteration()

        # learn from file
        num_iterations = tf.placeholder(dtype=tf.int64, shape=(), name='num-iterations')
        assert 'num_iterations' not in Module.placeholders
        Module.placeholders['num_iterations'] = num_iterations
        filenames = tf.placeholder(dtype=tf.string, shape=(None,), name='filenames')
        assert 'filenames' not in Module.placeholders
        Module.placeholders['filenames'] = filenames
        dataset = tf.data.TFRecordDataset(
            filenames=filenames, compression_type='GZIP', buffer_size=100000
            # num_parallel_reads=None
        )
        # dataset = dataset.cache(filename='')
        # filename: A tf.string scalar tf.Tensor, representing the name of a directory on the filesystem to use for caching tensors in this Dataset. If a filename is not provided, the dataset will be cached in memory.
        dataset = dataset.shuffle(buffer_size=100000, seed=None, reshuffle_each_iteration=True)
        dataset = dataset.repeat(count=None)
        features = {  # critically assumes max one trainable label
            label: value.feature() for value in self.values for label in value.trainable_labels()
        }
        # better performance after batch
        # dataset = dataset.map(
        #     map_func=(lambda serialized: tf.parse_single_example(
        #         serialized=serialized, features=features, name=None, example_names=None
        #     )), num_parallel_calls=None
        # )
        dataset = dataset.batch(batch_size=self.batch_size)  # drop_remainder=False
        dataset = dataset.map(
            map_func=(lambda serialized: tf.parse_example(
                serialized=serialized, features=features, name=None, example_names=None
            )), num_parallel_calls=None
        )
        dataset = dataset.prefetch(buffer_size=1)
        self.iterator = dataset.make_initializable_iterator(shared_name=None)

        def cond(iteration, losses, loss):
            return iteration < num_iterations

        def body(iteration, losses, loss):
            losses, loss, optimized = self.train_iteration(feed=self.iterator.get_next())
            with tf.control_dependencies(control_inputs=(optimized, loss)):
                iteration += 1
            return iteration, losses, loss

        losses, loss, optimized = self.train_iteration(feed=self.iterator.get_next())
        with tf.control_dependencies(control_inputs=(optimized,)):
            iteration = tf.constant(value=1, dtype=tf.int64, shape=(), verify_shape=False)
            self.optimized_fromfile, self.losses_fromfile, self.loss_fromfile = tf.while_loop(
                cond=cond, body=body, loop_vars=(iteration, losses, loss)
            )

        # synthesize
        num_synthesize = tf.placeholder(dtype=tf.int64, shape=(), name='num-synthesize')
        assert 'num_synthesize' not in Module.placeholders
        Module.placeholders['num_synthesize'] = num_synthesize
        x = self.encoding.sample(n=num_synthesize)
        x = self.customized_synthesize(x=x)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None)
        self.synthesized = dict()
        for value, x in zip(self.values, xs):
            xs = value.output_tensors(x=x)
            for label, x in xs.items():
                self.synthesized[label] = x

        # transform
        xs = list()
        for value in self.values:
            for label in value.trainable_labels():
                x = value.input_tensor()
                xs.append(x)
        x = tf.concat(values=xs, axis=1, name=None)
        x = self.encoder.transform(x=x)
        x = self.encoding.encode(x=x)
        x = self.customized_transform(x=x)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        self.transformed = dict()
        for value, x in zip(self.values, xs):
            xs = value.output_tensors(x=x)
            for label, x in xs.items():
                self.transformed[label] = x

    def learn(self, num_iterations, data=None, filenames=None, verbose=0):
        if (data is None) is (filenames is None):
            raise NotImplementedError

        # TODO: increment global step
        if filenames is None:
            data = data.copy()
            for value in self.values:
                data = value.preprocess(data=data)
            num_data = len(data)
            data = {
                label: data[label].get_values() for value in self.values
                for label in value.trainable_labels()
            }
            fetches = dict(self.losses)
            fetches['loss'] = self.loss
            fetches['optimized'] = self.optimized
            for iteration in range(num_iterations):
                batch = np.random.randint(num_data, size=self.batch_size)
                feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                fetched = self.run(fetches=fetches, feed_dict=feed_dict)
                if verbose > 0 and iteration % verbose + 1 == verbose:
                    print('{iteration}: {loss:1.2e}  ({losses})'.format(
                        iteration=(iteration + 1), loss=fetched['loss'], losses=', '.join(
                            '{name}: {loss}'.format(name=name, loss=fetched[name])
                            for name in self.losses
                        )
                    ))

        else:
            fetches = self.iterator.initializer
            feed_dict = dict(filenames=filenames)
            self.run(fetches=fetches, feed_dict=feed_dict)
            fetches = dict(self.losses_fromfile)
            fetches['loss'] = self.loss_fromfile
            fetches['optimized'] = self.optimized_fromfile
            if verbose == 0:
                feed_dict = dict(num_iterations=num_iterations)
                fetched = self.run(fetches=fetches, feed_dict=feed_dict)
            else:
                assert num_iterations % verbose == 0
                for iteration in range(num_iterations // verbose):
                    feed_dict = dict(num_iterations=verbose)
                    fetched = self.run(fetches=fetches, feed_dict=feed_dict)
                    print('{iteration}: {loss:1.2e}  ({losses})'.format(
                        iteration=((iteration + 1) * verbose), loss=fetched['loss'],
                        losses=', '.join(
                            '{name}: {loss}'.format(name=name, loss=fetched[name])
                            for name in self.losses
                        )
                    ))

    def synthesize(self, n):
        fetches = self.synthesized
        feed_dict = {'num_synthesize': n}
        synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
        synthesized = pd.DataFrame.from_dict(synthesized)
        for value in self.values:
            synthesized = value.postprocess(data=synthesized)
        return synthesized

    def transform(self, X, **transform_params):
        assert not transform_params
        for value in self.values:
            X = value.preprocess(data=X)
        fetches = self.transformed
        feed_dict = {
            label: X[label].get_values() for value in self.values
            for label in value.trainable_labels()
        }
        transformed = self.run(fetches=fetches, feed_dict=feed_dict)
        transformed = pd.DataFrame.from_dict(transformed)
        for value in self.values:
            transformed = value.postprocess(data=transformed)
        return transformed
