import logging
import time
from random import randrange
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import tensorflow as tf

from ..common import Synthesizer
from ..common.generative import SeriesEngine
from ..common.learning_manager import LearningManager
from ..common.util import record_summaries_every_n_global_steps
from ..common.values import DataFrameValue, Value, ValueExtractor
from ..config import SeriesConfig
from ..metadata.data_frame_meta import DataFrameMeta
from ..model import DataFrameModel
from ..model.factory import ModelFactory
from ..transformer import DataFrameTransformer

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
        model_factory = ModelFactory()
        self.df_meta = df_meta
        self.df_model: DataFrameModel = model_factory(df_meta)
        self.df_value: DataFrameValue = ValueExtractor.extract(
            df_meta=self.df_model, name='data_frame_value', config=config.value_factory_config
        )
        self.df_transformer: DataFrameTransformer = DataFrameTransformer.from_meta(self.df_model)
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
            name='vae', values=self.df_value.values(),
            identifier_label=self.df_meta.id_index,
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

    def get_data_feed_dict(self, df: pd.DataFrame) -> Dict[str, Sequence[tf.Tensor]]:
        data: Dict[str, Sequence[tf.Tensor]] = {
            name: tuple([
                tf.constant(df[column])
                for column in value.columns()
            ])
            for name, value in self.df_value.items()
        }
        return data

    def get_all_values(self) -> List[Value]:
        return list(self.df_value.values())

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
        raise NotImplementedError

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

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.df_transformer.transform(df)
        return df

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[Synthesizer, int, dict], bool] = None,
        callback_freq: int = 0, print_status_freq: int = 10, timeout: int = 2500
    ) -> None:

        t_start = time.time()

        assert num_iterations or self.learning_manager, "'num_iterations' must be set if learning_manager=False"

        df_train = df_train.copy()
        df_train = self.df_transformer.transform(df_train, max_workers=None).reset_index()

        groups, num_data = self.get_groups_feed_dict(df_train)
        max_seq_len = min(min(num_data), self.max_seq_len)

        with record_summaries_every_n_global_steps(callback_freq, self._global_step):
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
                    if self._writer is not None and iteration == 1:
                        tf.summary.trace_on(graph=True, profiler=False)
                        self.engine.learn(xs=feed_dict)
                        tf.summary.trace_export(name="Learn", step=self._global_step)
                        tf.summary.trace_off()
                    else:
                        self.engine.learn(xs=feed_dict)

                    # TODO: Implement callback/learning manager for series
                else:
                    self.engine.learn(xs=feed_dict)

                if print_status_freq > 0 and iteration % print_status_freq == 0:
                    self._print_learn_stats(self.get_losses(feed_dict), iteration)

                # Increment iteration number, and check if we reached max num_iterations
                iteration += 1
                if num_iterations:
                    keep_learning = iteration < num_iterations

                self._global_step.assign_add(1)

                if time.time() - t_start >= timeout:
                    break

    def _print_learn_stats(self, fetched, iteration):
        print('ITERATION {iteration} :: Loss: total={loss:.4f} ({losses})'.format(
            iteration=(iteration), loss=fetched['total-loss'], losses=', '.join(
                '{name}={loss:.4f}'.format(name=name, loss=loss)
                for name, loss in fetched.items()
            )
        ))

    def synthesize(
            self, num_rows: int, produce_nans: bool = False,
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
        raise NotImplementedError

    def encode(self, df_encode: pd.DataFrame, n_forecast: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError
