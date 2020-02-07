from typing import Callable, List, Union, Dict, Iterable, Optional,  Tuple
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import Value, ValueFactory, TypeOverride
from ..common.util import ProfilerArgs, record_summaries_every_n_global_steps
from ..common import Synthesizer
from ..common.generative import SeriesVAE

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class SeriesSynthesizer(Synthesizer):
    def __init__(
        self, df: pd.DataFrame, summarizer_dir: str = None, summarizer_name: str = None,
        profiler_args: ProfilerArgs = None,
        type_overrides: Dict[str, TypeOverride] = None,
        produce_nans_for: Union[bool, Iterable[str], None] = None,
        column_aliases: Dict[str, str] = None,
        # VAE latent space
        latent_size: int = 32,
        # Network
        network: str = 'resnet', capacity: int = 128, num_layers: int = 2,
        residual_depths: Union[None, int, List[int]] = 6,
        batchnorm: bool = True, activation: str = 'relu', dropout: Optional[float] = 0.2,
        # Optimizer
        optimizer: str = 'adam', learning_rate: float = 3e-3, decay_steps: int = None, decay_rate: float = None,
        initial_boost: int = 0, clip_gradients: float = 1.0,
        # Batch size
        batch_size: int = 64, increase_batch_size_every: Optional[int] = 500, max_batch_size: Optional[int] = 1024,
        # Losses
        beta: float = 1.0, weight_decay: float = 1e-6,
        # Categorical
        categorical_weight: float = 3.5, temperature: float = 1.0, moving_average: bool = True,
        # Continuous
        continuous_weight: float = 5.0,
        # Nan
        nan_weight: float = 1.0,
        # Conditions
        condition_columns: List[str] = None,
        # Person
        title_label: str = None, gender_label: str = None, name_label: str = None, firstname_label: str = None,
        lastname_label: str = None, email_label: str = None,
        mobile_number_label: str = None, home_number_label: str = None, work_number_label: str = None,
        # Bank
        bic_label: str = None, sort_code_label: str = None, account_label: str = None,
        # Address
        postcode_label: str = None, county_label: str = None, city_label: str = None, district_label: str = None,
        street_label: str = None, house_number_label: str = None, flat_label: str = None, house_name_label: str = None,
        address_label: str = None, postcode_regex: str = None,
        # Identifier
        identifier_label: str = None,
        # Rules to look for
        find_rules: Union[str, List[str]] = None,
        # SeriesSynthesizer
        encoding: str = 'variational',
        lstm_mode: int = 1,
        max_seq_len: int = 1024,
        condition_labels: List[str] = []
    ):
        super(SeriesSynthesizer, self).__init__(
            name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name
        )
        self.value_factory = ValueFactory(
            name='value_factory', df=df,
            capacity=capacity,
            continuous_weight=continuous_weight, categorical_weight=categorical_weight, temperature=temperature,
            moving_average=moving_average, nan_weight=nan_weight,
            type_overrides=type_overrides, produce_nans_for=produce_nans_for, column_aliases=column_aliases,
            condition_columns=condition_columns, find_rules=find_rules,
            # Person
            title_label=title_label, gender_label=gender_label, name_label=name_label, firstname_label=firstname_label,
            lastname_label=lastname_label, email_label=email_label, mobile_number_label=mobile_number_label,
            home_number_label=home_number_label, work_number_label=work_number_label,
            # Bank
            bic_label=bic_label, sort_code_label=sort_code_label, account_label=account_label,
            # Address
            postcode_label=postcode_label, county_label=county_label, city_label=city_label,
            district_label=district_label, street_label=street_label, house_number_label=house_number_label,
            flat_label=flat_label, house_name_label=house_name_label, address_label=address_label,
            postcode_regex=postcode_regex,
            # Identifier
            identifier_label=identifier_label,
        )
        self.encoding_type = encoding
        if lstm_mode not in (1, 2):
            raise NotImplementedError
        self.lstm_mode = lstm_mode
        self.max_seq_len = max_seq_len

        self.batch_size = batch_size
        self.tf_batch_size = tf.Variable(initial_value=batch_size, dtype=tf.int64)
        self.increase_batch_size_every = increase_batch_size_every
        self.max_batch_size: int = max_batch_size if max_batch_size else batch_size

        # VAE
        self.vae = SeriesVAE(
            name='vae', values=self.get_values(), conditions=self.get_conditions(),
            identifier_label=self.value_factory.identifier_label, identifier_value=self.value_factory.identifier_value,
            lstm_mode=self.lstm_mode, latent_size=latent_size,
            # network=network, capacity=capacity, num_layers=num_layers, residual_depths=residual_depths,
            # batchnorm=batchnorm, activation=activation,
            network='mlp', capacity=capacity, num_layers=1, residual_depths=None,
            batchnorm=False, activation='linear',
            dropout=dropout, optimizer=optimizer, learning_rate=tf.constant(learning_rate, dtype=tf.float32),
            decay_steps=decay_steps, decay_rate=decay_rate, initial_boost=initial_boost, clip_gradients=clip_gradients,
            beta=beta, weight_decay=weight_decay, summarize=(summarizer_dir is not None)
        )

        # Input argument placeholder for num_rows
        self.num_rows: Optional[tf.Tensor] = None

        # Output DF
        self.synthesized: Optional[Dict[str, tf.Tensor]] = None

    def get_values(self) -> List[Value]:
        return self.value_factory.get_values()

    def get_conditions(self) -> List[Value]:
        return self.value_factory.get_conditions()

    def get_losses(self, data: Dict[str, tf.Tensor]) -> tf.Tensor:
        self.vae.xs = data
        self.vae.loss()
        return self.vae.losses

    def specification(self):
        spec = super().specification()
        spec.update(
            values=[value.specification() for value in self.value_factory.get_values()],
            conditions=[value.specification() for value in self.value_factory.get_conditions()],
            vae=self.vae.specification(), batch_size=self.batch_size
        )
        return spec

    def preprocess(self, df):
        df = self.value_factory.preprocess(df)
        return df

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[Synthesizer, int, dict], bool] = Synthesizer.logging,
        callback_freq: int = 0, print_status_freq: int = 25
    ) -> None:

        assert num_iterations is not None and num_iterations > 0

        df_train = df_train.copy()
        df_train = self.value_factory.preprocess(df_train)

        groups, num_data = self.value_factory.get_groups_feed_dict(df_train)

        import time
        t_start = time.perf_counter()
        with record_summaries_every_n_global_steps(callback_freq, self.global_step):
            keep_learning = True
            iteration = 1
            while keep_learning:

                feed_dict = self.value_factory.get_group_feed_dict(groups, num_data, max_seq_len=self.max_seq_len)

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

                    if callback(self, iteration, self.vae.losses) is True:
                        return
                else:
                    self.vae.learn(xs=feed_dict)

                if iteration % print_status_freq == 0:
                    self._print_learn_stats(self.get_losses(feed_dict), iteration)
                    print(time.perf_counter() - t_start)

                # Increment iteration number, and check if we reached max num_iterations
                iteration += 1
                if num_iterations:
                    keep_learning = iteration < num_iterations

                self.global_step.assign_add(1)

        # return [value_data[batch] for value_data in data.values()], synth

    def _print_learn_stats(self, fetched, iteration):
        print('ITERATION {iteration} :: Loss: total={loss:.4f} ({losses})'.format(
            iteration=(iteration), loss=fetched['total-loss'], losses=', '.join(
                '{name}={loss:.4f}'.format(name=name, loss=loss)
                for name, loss in fetched.items()
            )
        ))

    def synthesize(self, series_length: int,
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

        elif series_lengths is not None:
            assert series_length is None, "Parameter 'series_length' is incompatible with 'series_lengths'."
            assert num_series is None or num_series == len(series_lengths)

        else:
            raise ValueError("Both 'num_series' and 'series_lengths' are None. One or the other is require to"
                             "synthesize data.")

        if self.lstm_mode == 0:
            raise NotImplementedError

        df_conditions = self.value_factory.preprocess_conditions(conditions=conditions)
        columns = self.value_factory.get_column_names()
        if self.value_factory.identifier_value:
            num_identifiers = self.value_factory.identifier_value.num_identifiers
        else:
            num_identifiers = 1

        feed_dict = self.value_factory.get_conditions_feed_dict(df_conditions, series_length, batch_size=None)
        synthesized = None

        if num_series is not None and series_length is not None:
            for i in range(num_series):
                other = self.vae.synthesize(tf.constant(series_length, dtype=tf.int64), cs=feed_dict)
                if self.value_factory.identifier_label:
                    other[self.value_factory.identifier_label] = tf.tile([i % num_identifiers], [series_length])
                other = pd.DataFrame.from_dict(other)[columns]
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        elif series_lengths is not None:
            for i, series_length in enumerate(series_lengths):
                other = self.vae.synthesize(tf.constant(series_length, dtype=tf.int64), cs=feed_dict)
                if self.value_factory.identifier_label:
                    other[self.value_factory.identifier_label] = tf.tile([i % num_identifiers], [series_length])
                other = pd.DataFrame.from_dict(other)[columns]
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        df_synthesized = pd.DataFrame.from_dict(synthesized)[columns]
        df_synthesized = self.value_factory.postprocess(df=df_synthesized)

        return df_synthesized

    def encode(self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:

        if conditions is not None:
            raise NotImplementedError

        columns = self.value_factory.get_column_names()
        df_encode = df_encode.copy()
        df_encode = self.value_factory.preprocess(df_encode)

        groups, num_data = self.value_factory.get_groups_feed_dict(df_encode)

        encoded, decoded = None, None

        for i in range(len(groups)):
            feed_dict = self.value_factory.get_group_feed_dict(groups, num_data, group=i)
            identifier = feed_dict[self.value_factory.identifier_label][0]
            encoded_i, decoded_i = self.vae.encode(xs=feed_dict, cs=dict())
            if self.value_factory.identifier_label:
                decoded_i[self.value_factory.identifier_label] = tf.tile([identifier], [num_data[i]])
            if not encoded or not decoded:
                encoded, decoded = encoded_i, decoded_i
            else:
                for k in encoded.keys():
                    encoded[k] = tf.concat((encoded[k], encoded_i[k]), axis=0)
                for k in decoded.keys():
                    decoded[k] = tf.concat((decoded[k], decoded_i[k]), axis=0)

        if not decoded or not encoded:
            return pd.DataFrame(), pd.DataFrame()

        df_synthesized = pd.DataFrame.from_dict(decoded)[columns]
        df_synthesized = self.value_factory.postprocess(df=df_synthesized)

        assert len(df_synthesized.columns) == len(columns)
        df_synthesized = df_synthesized[columns]

        # print(encoded)
        if self.lstm_mode == 1:
            latent = np.concatenate((encoded['sample'], encoded['mean'], encoded['std']), axis=1)
        else:
            print(encoded)
            len_sample = encoded['sample'].shape[0]
            # print(len_sample)
            latent = np.concatenate((
                encoded['sample'],
                tf.tile(encoded['mean'], [len_sample, 1]),
                tf.tile(encoded['std'], [len_sample, 1]),
            ), axis=1)
        df_encoded = pd.DataFrame.from_records(
            latent, columns=[f"{ls}_{n}" for ls in 'lms' for n in range(encoded['sample'].shape[1])])

        return df_encoded,  df_synthesized
