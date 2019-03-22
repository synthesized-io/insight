from random import randrange

import numpy as np
import pandas as pd

from .basic_synthesizer import BasicSynthesizer


class SeriesSynthesizer(BasicSynthesizer):

    def specification(self):
        spec = super().specification()
        # spec.update()
        return spec

    def learn(self, num_iterations=2500, data=None, filenames=None, verbose=0):
        if self.lstm_mode == 0 or self.identifier_label is None:
            raise NotImplementedError

        if (data is None) is (filenames is None):
            raise NotImplementedError

        if filenames is None:
            data = self.preprocess(data=data.copy())

            groups = [group[1] for group in data.groupby(by=self.identifier_label)]
            num_data = [len(group) for group in groups]
            for n, group in enumerate(groups):
                groups[n] = {
                    label: group[label].get_values() for value in self.values
                    for label in value.input_labels()
                }

            fetches = self.optimized
            if verbose > 0:
                verbose_fetches = dict(self.losses)
                verbose_fetches['loss'] = self.loss

            for iteration in range(num_iterations):
                group = randrange(len(num_data))
                data = groups[group]
                start = randrange(max(num_data[group] - self.batch_size, 1))
                batch = np.arange(start, min(start + self.batch_size, num_data[group]))

                feed_dict = {label: value_data[batch] for label, value_data in data.items()}
                self.run(fetches=fetches, feed_dict=feed_dict)

                if verbose > 0 and (
                    iteration == 0 or iteration + 1 == verbose // 2 or
                    iteration % verbose + 1 == verbose
                ):
                    group = randrange(len(num_data))
                    data = groups[group]
                    start = randrange(max(num_data[group] - 1024, 1))
                    batch = np.arange(start, min(start + 1024, num_data[group]))

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
            self.run(fetches=fetches, feed_dict=feed_dict)
            # assert num_iterations % verbose == 0
            # for iteration in range(num_iterations // verbose):
            #     feed_dict = dict(num_iterations=verbose)
            #     fetched = self.run(fetches=fetches, feed_dict=feed_dict)
            #     self.log_metrics(data, fetched, iteration)

    def synthesize(self, num_series=None, series_length=None, series_lengths=None):
        # Either num_series and series_length, or series_lenghts, or ???
        if self.lstm_mode == 0 or self.identifier_label is None:
            raise NotImplementedError

        fetches = self.synthesized

        if num_series is not None:
            assert series_length is not None and series_lengths is None
            feed_dict = {'num_synthesize': series_length}
            columns = [label for value in self.values for label in value.output_labels()]
            synthesized = None
            for _ in range(num_series):
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                other = pd.DataFrame.from_dict(other)[columns]
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        elif series_lengths is not None:
            assert series_length is None
            columns = [label for value in self.values for label in value.output_labels()]
            synthesized = None
            for series_length in series_lengths:
                feed_dict = {'num_synthesize': series_length}
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                other = pd.DataFrame.from_dict(other)[columns]
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        for value in self.values:
            synthesized = value.postprocess(data=synthesized)

        return synthesized
