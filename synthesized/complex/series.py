import logging
import random
import time
from random import randrange
from typing import Callable, List, Union, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import Synthesizer
from ..common.values import Value, ValueFactory
from ..common.generative import SeriesVAE
from ..common.learning_manager import LearningManager
from ..common.util import record_summaries_every_n_global_steps
from ..config import SeriesConfig
from ..metadata import DataFrameMeta, ValueMeta

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class SeriesSynthesizer(Synthesizer):
    def __init__(
        self, df_meta: DataFrameMeta, summarizer_dir: str = None, summarizer_name: str = None,
        config: SeriesConfig = SeriesConfig(),
        # SeriesSynthesizer
        condition_labels: List[str] = None,
        # Evaluation conditions
        learning_manager: bool = False
    ):
        super(SeriesSynthesizer, self).__init__(
            name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name
        )
        self.df_meta = df_meta
        self.value_factory = ValueFactory(
            df_meta=df_meta, name='value_factory', conditions=condition_labels, config=config.value_factory_config
        )
        if config.lstm_mode not in ('lstm', 'vrae', 'rdssm'):
            raise NotImplementedError
        self.lstm_mode = config.lstm_mode

        # if identifier_label:
        #     min_len = df.groupby(identifier_label).count().min().values[0]
        #     max_seq_len = min(max_seq_len, min_len)

        self.max_seq_len = config.max_seq_len

        self.batch_size = config.batch_size
        self.tf_batch_size = tf.Variable(initial_value=config.batch_size, dtype=tf.int64)
        self.increase_batch_size_every = config.increase_batch_size_every
        self.max_batch_size: int = config.max_batch_size if config.max_batch_size else config.batch_size

        # VAE
        self.vae = SeriesVAE(
            name='vae', values=self.get_values(), conditions=self.get_conditions(),
            identifier_label=self.df_meta.id_index, identifier_value=self.value_factory.identifier_value,
            encoding=self.lstm_mode, latent_size=config.latent_size,
            network=config.network, capacity=config.capacity, num_layers=config.num_layers,
            residual_depths=config.residual_depths,
            batch_norm=config.batch_norm, activation=config.activation, series_dropout=config.series_dropout,
            optimizer=config.optimizer, learning_rate=tf.constant(config.learning_rate, dtype=tf.float32),
            decay_steps=config.decay_steps, decay_rate=config.decay_rate, initial_boost=config.initial_boost,
            clip_gradients=config.clip_gradients,
            beta=config.beta, weight_decay=config.weight_decay
        )

        # Input argument placeholder for num_rows
        self.num_rows: Optional[tf.Tensor] = None

        # Output DF
        self.synthesized: Optional[Dict[str, tf.Tensor]] = None

        # Learning Manager
        self.learning_manager: Optional[LearningManager] = None
        if learning_manager:
            self.use_vae_loss = True
            self.learning_manager = LearningManager()
            self.learning_manager.set_check_frequency(self.batch_size)
            raise NotImplementedError

    def get_values(self) -> List[Value]:
        return self.value_factory.get_values()

    def get_conditions(self) -> List[Value]:
        return self.value_factory.get_conditions()

    def get_all_values(self) -> List[Value]:
        return self.value_factory.all_values

    def get_value_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return [(v, self.df_meta[v.name]) for v in self.value_factory.all_values]

    def get_condition_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return [(v, self.df_meta[v.name]) for v in self.value_factory.get_conditions()]

    def get_losses(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        self.vae.xs = data
        self.vae.loss()
        return self.vae.losses

    def get_data_feed_dict(self, df: pd.DataFrame) -> Dict[str, List[tf.Tensor]]:
        data = {
            value.name: [tf.constant(df[name].to_numpy(), dtype=value.dtype) for name in meta.learned_input_columns()]
            for value, meta in self.get_value_meta_pairs()
        }
        return data

    def get_groups_feed_dict(self, df: pd.DataFrame) -> Tuple[List[Dict[str, List[tf.Tensor]]], List[int]]:
        if self.df_meta.id_index is None:
            num_data = [len(df)]
            groups = [{
                value.name: [tf.constant(df[name].to_numpy(), dtype=value.dtype)
                             for name in meta.learned_input_columns()]
                for value, meta in self.get_value_meta_pairs()
            }]

        else:
            groups = [group[1] for group in df.groupby(by=self.df_meta.id_index)]
            num_data = [len(group) for group in groups]
            for n in range(len(groups)):
                groups[n] = {
                    value.name: [tf.constant(groups[n][name], dtype=value.dtype)
                                 for name in meta.learned_input_columns()]
                    for value, meta in self.get_value_meta_pairs()
                }

        return groups, num_data

    def get_group_feed_dict(self, groups, num_data, max_seq_len=None, group=None) -> Dict[str, List[tf.Tensor]]:
        group = group if group is not None else randrange(len(num_data))
        data = groups[group]

        if max_seq_len and num_data[group] > max_seq_len:
            start = randrange(num_data[group] - max_seq_len)
            batch = tf.range(start, start + max_seq_len)
        else:
            batch = tf.range(num_data[group])

        feed_dict = {name: [tf.nn.embedding_lookup(params=val, ids=batch) for val in value_data]
                     for name, value_data in data.items()}

        return feed_dict

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            values=[value.specification() for value in self.value_factory.get_values()],
            conditions=[value.specification() for value in self.value_factory.get_conditions()],
            vae=self.vae.specification(), batch_size=self.batch_size
        )
        return spec

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.df_meta.preprocess(df)
        return df

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[Synthesizer, int, dict], bool] = Synthesizer.logging,
        callback_freq: int = 0, print_status_freq: int = 10, timeout: int = 2500
    ) -> None:

        t_start = time.time()

        assert num_iterations or self.learning_manager, "'num_iterations' must be set if learning_manager=False"

        df_train = df_train.copy()
        df_train = self.df_meta.preprocess(df_train)

        groups, num_data = self.get_groups_feed_dict(df_train)
        max_seq_len = min(min(num_data), self.max_seq_len)

        with record_summaries_every_n_global_steps(callback_freq, self.global_step):
            keep_learning = True
            iteration = 1
            while keep_learning:

                feed_dicts = [self.get_group_feed_dict(groups, num_data, max_seq_len=max_seq_len)
                              for _ in range(self.batch_size)]

                # TODO: Code below will fail if sequences don't have same shape.
                feed_dict = {
                    value.name: [
                        tf.stack([fd[value.name][n] for fd in feed_dicts], axis=0)
                        for n, name in enumerate(self.df_meta[value.name].learned_input_columns())
                    ]
                    for value in self.get_all_values()
                }

                if callback is not None and callback_freq > 0 and (
                    iteration == 1 or iteration == num_iterations or iteration % callback_freq == 0
                ):
                    if self.writer is not None and iteration == 1:
                        tf.summary.trace_on(graph=True, profiler=False)
                        self.vae.learn(xs=feed_dict)
                        tf.summary.trace_export(name="Learn", step=self.global_step)
                        tf.summary.trace_off()
                    else:
                        self.vae.learn(xs=feed_dict)

                    # if callback(self, iteration, self.vae.losses) is True:
                    #     return
                else:
                    self.vae.learn(xs=feed_dict)

                if print_status_freq > 0 and iteration % print_status_freq == 0:
                    self._print_learn_stats(self.get_losses(feed_dict), iteration)

                # Increment iteration number, and check if we reached max num_iterations
                iteration += 1
                if num_iterations:
                    keep_learning = iteration < num_iterations

                self.global_step.assign_add(1)

                if time.time() - t_start >= timeout:
                    break

        # return [value_data[batch] for value_data in data.values()], synth

    def _print_learn_stats(self, fetched, iteration):
        print('ITERATION {iteration} :: Loss: total={loss:.4f} ({losses})'.format(
            iteration=(iteration), loss=fetched['total-loss'], losses=', '.join(
                '{name}={loss:.4f}'.format(name=name, loss=loss)
                for name, loss in fetched.items()
            )
        ))

    def synthesize(self, series_length: int = None,
                   conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None,
                   num_series: int = None, series_lengths: List[int] = None
                   ) -> pd.DataFrame:
        """Synthesize a dataset from the learned model

        :param series_length: Length of the synthesized series, if all have same length.
        :param conditions: Conditions.
        :param progress_callback: Progress bar callback.
        :param num_series: Number of series to synthesize.
        :param series_lengths: List of lenghts of each synthesized series, if they are different
        :return: Synthesized dataframe.
        """
        if num_series is not None:
            assert series_length is not None, "If 'num_series' is given, 'series_length' must be defined."
            assert series_lengths is None, "Parameter 'series_lengths' is incompatible with 'num_series'."

            series_lengths = [series_length] * num_series

        elif series_lengths is not None:
            assert series_length is None, "Parameter 'series_length' is incompatible with 'series_lengths'."
            assert num_series is None or num_series == len(series_lengths)

            num_series = len(series_lengths)

        else:
            raise ValueError("Both 'num_series' and 'series_lengths' are None. One or the other is require to"
                             "synthesize data.")

        df_conditions = self.df_meta.preprocess_by_name(conditions, [c.name for c in self.get_conditions()])
        columns = self.df_meta.columns

        feed_dict = self.get_conditions_feed_dict(df_conditions, series_length, batch_size=None)
        synthesized = None

        # Get identifiers to iterate
        if self.value_factory.identifier_value and num_series > self.value_factory.identifier_value.num_identifiers:
            raise ValueError("Number of series to synthesize is bigger than original dataset.")

        print(f'num_series: {num_series}, series_lengths: {series_length}')

        with record_summaries_every_n_global_steps(0, self.global_step):
            for identifier in random.sample(range(num_series), num_series):
                print('synthesizing series.')
                series_length = series_lengths[identifier]
                tf_identifier = tf.constant([identifier])
                other = self.vae.synthesize(tf.constant(series_length, dtype=tf.int64), cs=feed_dict,
                                            identifier=tf_identifier)
                other = self.df_meta.split_outputs(other)
                other = pd.DataFrame.from_dict(other)[columns]
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        df_synthesized = pd.DataFrame.from_dict(synthesized)[columns]
        df_synthesized = self.df_meta.postprocess(df=df_synthesized)

        return df_synthesized

    def encode(self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None,
               n_forecast: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if conditions is not None:
            raise NotImplementedError

        columns = self.df_meta.columns
        df_encode = df_encode.copy()
        df_encode = self.df_meta.preprocess(df_encode)

        groups, num_data = self.get_groups_feed_dict(df_encode)

        encoded, decoded = None, None

        for i in range(len(groups)):

            feed_dict = self.get_group_feed_dict(groups, num_data, group=i)
            encoded_i, decoded_i = self.vae.encode(xs=feed_dict, cs=dict(), n_forecast=n_forecast)
            if len(encoded_i['sample'].shape) == 1:
                encoded_i['sample'] = tf.expand_dims(encoded_i['sample'], axis=0)

            if self.df_meta.id_index:
                identifier = feed_dict[self.df_meta.id_index][0]
                decoded_i[self.df_meta.id_index] = tf.tile([identifier], [num_data[i] + n_forecast])

            if not encoded or not decoded:
                encoded, decoded = encoded_i, decoded_i
            else:
                for k in encoded.keys():
                    encoded[k] = tf.concat((encoded[k], encoded_i[k]), axis=0)
                for k in decoded.keys():
                    decoded[k] = tf.concat((decoded[k], decoded_i[k]), axis=0)

        if not decoded or not encoded:
            return pd.DataFrame(), pd.DataFrame()

        decoded = self.df_meta.split_outputs(decoded)
        df_synthesized = pd.DataFrame.from_dict(decoded)[columns]
        df_synthesized = self.df_meta.postprocess(df=df_synthesized)

        latent = np.concatenate((encoded['sample'], encoded['mean'], encoded['std']), axis=1)

        df_encoded = pd.DataFrame.from_records(
            latent, columns=[f"{ls}_{n}" for ls in ['l', 'm', 's'] for n in range(encoded['sample'].shape[1])])

        return df_encoded,  df_synthesized
