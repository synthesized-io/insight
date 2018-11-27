import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ks_2samp

from .encodings import encoding_modules
from .module import Module
from .optimizers import Optimizer
from .synthesizer import Synthesizer
from .transformations import transformation_modules
from .values import identify_value


class BasicSynthesizer(Synthesizer):

    def __init__(
        self, data, exclude_encoding_loss=False, summarizer=False,
        # architecture
        network='resnet', encoding='variational',
        # hyperparameters
        capacity=64, depth=4, learning_rate=3e-4, weight_decay=1e-5, batch_size=64,
        # person
        gender_label=None, name_label=None, firstname_label=None, lastname_label=None,
        email_label=None,
        # address
        postcode_label=None, street_label=None,
        # identifier
        identifier_label=None
    ):
        super().__init__(name='synthesizer', summarizer=summarizer)

        self.exclude_encoding_loss = exclude_encoding_loss

        self.network_type = network
        self.encoding_type = encoding
        self.capacity = capacity
        self.depth = depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
            if value is not None:
                value.extract(data=data)
                self.values.append(value)
                self.value_output_sizes.append(value.output_size())
                input_size += value.input_size()
                output_size += value.output_size()
            print(name, value)

        self.encoder = self.add_module(
            module=self.network_type, modules=transformation_modules, name='encoder',
            input_size=input_size, layer_sizes=[self.capacity for _ in range(self.depth)],
            weight_decay=self.weight_decay
        )

        self.encoding = self.add_module(
            module=self.encoding_type, modules=encoding_modules, name='encoding',
            input_size=self.encoder.size(), encoding_size=self.capacity
        )

        self.decoder = self.add_module(
            module=self.network_type, modules=transformation_modules, name='decoder',
            input_size=self.encoding.size(),
            layer_sizes=[self.capacity for _ in range(self.depth)], weight_decay=self.weight_decay
        )

        self.output = self.add_module(
            module='dense', modules=transformation_modules, name='output',
            input_size=self.decoder.size(), output_size=output_size, batchnorm=False,
            activation='none'
        )

        # https://twitter.com/karpathy/status/801621764144971776  ;-)
        self.optimizer = self.add_module(
            module=Optimizer, name='optimizer', algorithm='adam', learning_rate=self.learning_rate,
            clip_gradients=1.0
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            network=self.network_type, encoding=self.encoding_type, capacity=self.capacity,
            depth=self.depth, learning_rate=self.learning_rate, weight_decay=self.weight_decay,
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
        x = tf.concat(values=xs, axis=1)
        x = self.encoder.transform(x=x)
        x, encoding_loss = self.encoding.encode(x=x, encoding_loss=True)
        x = self.customized_transform(x=x)
        x = self.decoder.transform(x=x)
        x = self.output.transform(x=x)
        xs = tf.split(
            value=x, num_or_size_splits=self.value_output_sizes, axis=1, num=None, name=None
        )
        reg_losses = tf.losses.get_regularization_losses(scope=None)
        if len(reg_losses) > 0:
            reg_loss = tf.add_n(inputs=reg_losses)
            losses = dict(encoding=encoding_loss, regularization=reg_loss)
        else:
            losses = dict(encoding=encoding_loss)
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
        x = tf.concat(values=xs, axis=1)
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
            data = self.preprocess(data=data.copy())
            num_data = len(data)
            data = {
                label: data[label].get_values() for value in self.values
                for label in value.trainable_labels()
            }
            fetches = self.optimized
            if verbose > 0:
                verbose_fetches = dict(self.losses)
                verbose_fetches['loss'] = self.loss
            for iteration in range(num_iterations):
                batch = np.random.randint(num_data, size=self.batch_size)
                feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                self.run(fetches=fetches, feed_dict=feed_dict, summarize=True)
                do_logging = iteration == 0 or iteration + 1 == verbose // 2 or \
                    iteration % verbose + 1 == verbose
                if verbose > 0 and do_logging:
                    batch = np.random.randint(num_data, size=1024)
                    feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                    fetched = self.run(fetches=verbose_fetches, feed_dict=feed_dict)
                    self.log_metrics(data, fetched, iteration)

        else:
            if verbose > 0:
                raise NotImplementedError
            fetches = self.iterator.initializer
            feed_dict = dict(filenames=filenames)
            self.run(fetches=fetches, feed_dict=feed_dict)
            fetches = self.optimized_fromfile
            feed_dict = dict(num_iterations=num_iterations)
            self.run(fetches=fetches, feed_dict=feed_dict, summarize=True)
            # assert num_iterations % verbose == 0
            # for iteration in range(num_iterations // verbose):
            #     feed_dict = dict(num_iterations=verbose)
            #     fetched = self.run(fetches=fetches, feed_dict=feed_dict, summarize=True)
            #     self.log_metrics(data, fetched, iteration)

    def log_metrics(self, data, fetched, iteration):
        print('\niteration: {}'.format(iteration + 1))
        print('loss: total={loss:1.2e} ({losses})'.format(
            iteration=(iteration + 1), loss=fetched['loss'], losses=', '.join(
                '{name}={loss}'.format(name=name, loss=fetched[name])
                for name in self.losses
            )
        ))
        synthesized = self.synthesize(10000)
        synthesized = self.preprocess(data=synthesized)
        dist_by_col = [(col, ks_2samp(data[col], synthesized[col].get_values())[0]) for col in data.keys()]
        avg_dist = np.mean([dist for (col, dist) in dist_by_col])
        dists = ', '.join(['{col}={dist:.2f}'.format(col=col, dist=dist) for (col, dist) in dist_by_col])
        print('KS distances: avg={avg_dist:.2f} ({dists})'.format(avg_dist=avg_dist, dists=dists))

    def synthesize(self, n):
        fetches = self.synthesized
        feed_dict = {'num_synthesize': n % 1024}
        synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
        synthesized = pd.DataFrame.from_dict(synthesized)
        feed_dict = {'num_synthesize': 1024}
        for k in range(n // 1024):
            other = self.run(fetches=fetches, feed_dict=feed_dict)
            other = pd.DataFrame.from_dict(other)
            synthesized = synthesized.append(other, ignore_index=True)
        for value in self.values:
            synthesized = value.postprocess(data=synthesized)
        return synthesized

    def transform(self, X, **transform_params):
        assert not transform_params
        data = self.preprocess(data=X.copy())
        fetches = self.transformed
        feed_dict = {
            label: data[label].get_values() for value in self.values
            for label in value.trainable_labels()
        }
        transformed = self.run(fetches=fetches, feed_dict=feed_dict)
        transformed = pd.DataFrame.from_dict(transformed)
        for value in self.values:
            transformed = value.postprocess(data=transformed)
        return transformed
