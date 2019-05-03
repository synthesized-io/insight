from random import randrange

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .basic_synthesizer import BasicSynthesizer


class SeriesSynthesizer(BasicSynthesizer):

    def specification(self):
        spec = super().specification()
        # spec.update()
        return spec

    def learn(self, num_iterations=2500, data=None, filenames=None, verbose=0):
        if self.lstm_mode == 0 or (
            self.identifier_label is None and len(self.condition_labels) == 0
        ):
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

                feed_dict = dict()
                for label, value_data in data.items():
                    if label in self.condition_labels and self.lstm_mode == 2:
                        feed_dict[label] = value_data[:1]
                    else:
                        feed_dict[label] = value_data[batch]
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

    def log_metrics(self, data, fetched, iteration):
        print('\niteration: {}'.format(iteration + 1))
        print('loss: total={loss:1.2e} ({losses})'.format(
            iteration=(iteration + 1), loss=fetched['loss'], losses=', '.join(
                '{name}={loss}'.format(name=name, loss=fetched[name])
                for name in self.losses
            )
        ))
        self.loss_history.append({name: fetched[name] for name in self.losses})

        synthesized = self.synthesize(num_series=100, series_length=100)
        synthesized = self.preprocess(data=synthesized)
        dist_by_col = [(col, ks_2samp(data[col], synthesized[col].get_values())[0]) for col in data.keys() if col != self.identifier_label]
        avg_dist = np.mean([dist for (col, dist) in dist_by_col])
        dists = ', '.join(['{col}={dist:.2f}'.format(col=col, dist=dist) for (col, dist) in dist_by_col])
        print('KS distances: avg={avg_dist:.2f} ({dists})'.format(avg_dist=avg_dist, dists=dists))
        self.ks_distance_history.append(dict(dist_by_col))

    def synthesize(self, num_series=None, series_length=None, series_lengths=None, condition=None):
        # Either num_series and series_length, or series_lenghts, or ???
        if self.lstm_mode == 0 or (
            self.identifier_label is None and len(self.condition_labels) == 0
        ):
            raise NotImplementedError

        fetches = self.synthesized
        if condition is None:
            feed_dict = dict()
        else:
            feed_dict = dict(condition)

        if num_series is not None:
            assert series_length is not None and series_lengths is None
            feed_dict['num_synthesize'] = series_length
            columns = [
                label for value in self.values if value.name not in self.condition_labels
                for label in value.output_labels()
            ]
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
            columns = [
                label for value in self.values if value.name not in self.condition_labels
                for label in value.output_labels()
            ]
            synthesized = None
            for series_length in series_lengths:
                feed_dict['num_synthesize'] = series_length
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                other = pd.DataFrame.from_dict(other)[columns]
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        for value in self.values:
            if value.name not in self.condition_labels:
                synthesized = value.postprocess(data=synthesized)

        return synthesized
