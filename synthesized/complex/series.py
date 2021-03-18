# type: ignore
import logging
import random
import time
from random import randrange
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import Synthesizer
from ..common.generative import SeriesEngine
from ..common.learning_manager import LearningManager
from ..common.util import record_summaries_every_n_global_steps
from ..common.values import Value, ValueFactory
from ..config import SeriesConfig
from ..metadata.data_frame_meta import DataFrameMeta

logger = logging.getLogger(__name__)


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
        self.max_seq_len = config.max_seq_len

        self.batch_size = config.batch_size
        self.tf_batch_size = tf.Variable(initial_value=config.batch_size, dtype=tf.int64)
        self.increase_batch_size_every = config.increase_batch_size_every
        self.max_batch_size: int = config.max_batch_size if config.max_batch_size else config.batch_size

        # VAE
        self.engine = SeriesEngine(
            name='vae', values=self.get_values(), conditions=self.get_conditions(),
            identifier_label=self.df_meta.id_index_name, identifier_value=self.value_factory.identifier_value,
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
            self.use_engine_loss = True
            self.learning_manager = LearningManager()
            self.learning_manager.set_check_frequency(self.batch_size)
            raise NotImplementedError

    def get_values(self) -> List[Value]:
        return self.value_factory.get_values()

    def get_conditions(self) -> List[Value]:
        return self.value_factory.get_conditions()

    def get_data_feed_dict(self, df: pd.DataFrame) -> Dict[str, Sequence[tf.Tensor]]:
        data: Dict[str, Sequence[tf.Tensor]] = {
            value.name: tuple([
                tf.constant(df[name])
                for m_name in value.meta_names for name in self.df_meta[m_name].learned_input_columns()
            ])
            for value in self.get_all_values()
        }
        return data

    def get_conditions_feed_dict(self, df_conditions: pd.DataFrame) -> Dict[str, Sequence[tf.Tensor]]:
        data: Dict[str, Sequence[tf.Tensor]] = {
            value.name: tuple([
                tf.constant(df_conditions[name])
                for m_name in value.meta_names for name in self.df_meta[m_name].learned_input_columns()
            ])
            for value in self.get_conditions()
        }
        return data

    def get_all_values(self) -> List[Value]:
        return self.value_factory.all_values

    def get_losses(self, data: Dict[str, tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        if data is not None:
            self.engine.loss(data)
        losses = {
            'total-loss': self.engine.total_loss,
            'kl-loss': self.engine.kl_loss,
            'regularization-loss': self.engine.regularization_loss,
            'reconstruction-loss': self.engine.reconstruction_loss
        }
        return losses

    def get_groups_feed_dict(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Sequence[tf.Tensor]]], List[int]]:
        if self.df_meta.id_index_name is None:
            num_data = [len(df)]
            groups: List[Dict[str, Sequence[tf.Tensor]]] = [{
                value.name: tuple([
                    tf.constant(df[name])
                    for m_name in value.meta_names for name in self.df_meta[m_name].learned_input_columns()
                ])
                for value in self.get_all_values()
            }]

        else:
            groups = [group[1] for group in df.groupby(by=self.df_meta.id_index_name)]
            num_data = [len(group) for group in groups]
            for n in range(len(groups)):
                groups[n] = {
                    value.name: tuple([
                        tf.constant(groups[n][name])
                        for m_name in value.meta_names for name in (
                            self.df_meta[m_name].learned_input_columns()
                            if not isinstance(self.df_meta[m_name], IdentifierMeta) else [m_name]  # noqa: F821
                        )
                    ])
                    for value in self.get_all_values()
                }

        return groups, num_data

    def get_group_feed_dict(
            self, groups: List[Dict[str, Sequence[tf.Tensor]]], num_data, max_seq_len=None, group=None
    ) -> Dict[str, Sequence[tf.Tensor]]:
        group = group if group is not None else randrange(len(num_data))
        data = groups[group]

        if max_seq_len and num_data[group] > max_seq_len:
            start = randrange(num_data[group] - max_seq_len)
            batch = tf.range(start, start + max_seq_len)
        else:
            batch = tf.range(num_data[group])

        feed_dict: Dict[str, Sequence[tf.Tensor]] = {
            name: tuple([
                tf.nn.embedding_lookup(params=val, ids=batch)
                for val in values
            ])
            for name, values in data.items()
        }

        return feed_dict

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            values=[value.specification() for value in self.value_factory.get_values()],
            conditions=[value.specification() for value in self.value_factory.get_conditions()],
            engine=self.engine.specification(), batch_size=self.batch_size
        )
        return spec

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.df_meta.preprocess(df)
        return df

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[Synthesizer, int, dict], bool] = None,
        callback_freq: int = 0, print_status_freq: int = 10, timeout: int = 2500
    ) -> None:

        t_start = time.time()

        assert num_iterations or self.learning_manager, "'num_iterations' must be set if learning_manager=False"

        df_train = df_train.copy()
        df_train = self.df_meta.preprocess(df_train, max_workers=None).reset_index()

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
                        for n in range(len(feed_dicts[0][value.name]))
                    ]
                    for value in self.get_all_values()
                }

                if callback is not None and callback_freq > 0 and (
                    iteration == 1 or iteration == num_iterations or iteration % callback_freq == 0
                ):
                    if self.writer is not None and iteration == 1:
                        tf.summary.trace_on(graph=True, profiler=False)
                        self.engine.learn(xs=feed_dict)
                        tf.summary.trace_export(name="Learn", step=self.global_step)
                        tf.summary.trace_off()
                    else:
                        self.engine.learn(xs=feed_dict)

                    # if callback(self, iteration, self.engine.losses) is True:
                    #     return
                else:
                    self.engine.learn(xs=feed_dict)

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

    def synthesize(
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None, produce_nans: bool = False,
            progress_callback: Callable[[int], None] = None, num_series: int = 1
    ) -> pd.DataFrame:
        """Synthesize a dataset from the learned model

        Args:
            num_rows: Length of the synthesized series.
            num_series: Number of series to synthesize.
            conditions: Conditions.
            produce_nans: Whether to produce NaNs.
            progress_callback: Progress bar callback.

        Returns: Synthesized dataframe.
        """
        series_length = num_rows
        df_conditions = self.df_meta.preprocess_by_name(conditions, [c.name for c in self.get_conditions()])
        columns = self.df_meta.columns

        feed_dict = self.get_conditions_feed_dict(df_conditions)
        synthesized = None

        # Get identifiers to iterate
        if self.value_factory.identifier_value and num_series > self.value_factory.identifier_value.num_identifiers:
            raise ValueError("Number of series to synthesize is bigger than original dataset.")

        with record_summaries_every_n_global_steps(0, self.global_step):
            for identifier in random.sample(range(num_series), num_series):
                tf_identifier = tf.constant([identifier])
                other = self.engine.synthesize(tf.constant(series_length, dtype=tf.int64), cs=feed_dict,
                                               identifier=tf_identifier)
                other = self.df_meta.split_outputs(other)
                other = pd.DataFrame.from_dict(other)
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        df_synthesized = pd.DataFrame.from_dict(synthesized)
        df_synthesized = self.df_meta.postprocess(df=df_synthesized, max_workers=None)[columns]

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
            encoded_i, decoded_i = self.engine.encode(xs=feed_dict, cs=dict(), n_forecast=n_forecast)
            if len(encoded_i['sample'].shape) == 1:
                encoded_i['sample'] = tf.expand_dims(encoded_i['sample'], axis=0)

            if self.df_meta.id_index_name:
                identifier = feed_dict[self.df_meta.id_index_name]
                decoded_i[self.df_meta.id_index_name] = tf.tile([identifier], [num_data[i] + n_forecast])

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

        return df_encoded, df_synthesized
