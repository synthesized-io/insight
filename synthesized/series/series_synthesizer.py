from random import randrange
import enum
from collections import OrderedDict
from typing import Callable, List, Union, Dict, Set, Iterable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import Value, ValueFactory
from ..common.util import ProfilerArgs
from ..common.values import ContinuousValue
from ..common import Synthesizer


class TypeOverride(enum.Enum):
    ID = 'ID'
    DATE = 'DATE'
    CATEGORICAL = 'CATEGORICAL'
    CONTINUOUS = 'CONTINUOUS'
    ENUMERATION = 'ENUMERATION'


class SeriesSynthesizer(Synthesizer,  ValueFactory):
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
        batchnorm: bool = True, activation: str = 'relu',
        # Optimizer
        optimizer: str = 'adam', learning_rate: float = 3e-3, decay_steps: int = None, decay_rate: float = None,
        initial_boost: int = 0, clip_gradients: float = 1.0, batch_size: int = 1024,
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
        # Address
        postcode_label: str = None, city_label: str = None, street_label: str = None,
        address_label: str = None, postcode_regex: str = None,
        # Identifier
        identifier_label: str = None,
        # Rules to look for
        find_rules: Union[str, List[str]] = None,
        ## SeriesSynthesizer
        encoding: str = 'variational',
        lstm_mode: int = 1,
        max_seq_len: int = 1024,
        condition_labels: List[str] = ()
    ):
        Synthesizer.__init__(self, name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name,
                             profiler_args=profiler_args)
        if type_overrides is None:
            self.type_overrides: Dict[str, TypeOverride] = dict()
        else:
            self.type_overrides = type_overrides

        if isinstance(produce_nans_for, Iterable):
            self.produce_nans_for: Set[str] = set(produce_nans_for)
        elif produce_nans_for:
            self.produce_nans_for = set(df.columns)
        else:
            self.produce_nans_for = set()

        if column_aliases is None:
            self.column_aliases: Dict[str, str] = {}
        else:
            self.column_aliases = column_aliases

        if condition_columns is None:
            self.condition_columns: List[str] = []
        else:
            self.condition_columns = condition_columns

        if find_rules is None:
            self.find_rules: Union[str, List[str]] = []
        else:
            self.find_rules = find_rules

        self.encoding_type = encoding
        if lstm_mode not in (1, 2):
            raise NotImplementedError
        self.lstm_mode = lstm_mode
        self.max_seq_len = max_seq_len

        self.batch_size = batch_size

        # For identify_value (should not be necessary)
        self.capacity = capacity
        self.weight_decay = weight_decay

        # Categorical
        self.categorical_weight = categorical_weight
        self.temperature = temperature
        self.moving_average = moving_average

        # Continuous
        self.continuous_weight = continuous_weight

        # Nan
        self.nan_weight = nan_weight

        # Overall columns
        self.columns = list(df.columns)
        # Person
        self.person_value: Optional[Value] = None
        self.bank_value: Optional[Value] = None
        self.title_label = title_label
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        # Address
        self.address_value: Optional[Value] = None
        self.postcode_label = postcode_label
        self.address_label = address_label
        self.postcode_regex = postcode_regex
        # Identifier
        self.identifier_value: Optional[Value] = None
        self.identifier_label = identifier_label
        # Date
        self.date_value: Optional[Value] = None

        # history
        self.loss_history = list()
        self.ks_distance_history = list()

        # Values
        self.values: List[Value] = list()
        self.conditions: List[Value] = list()
        if identifier_label:
            assert identifier_label in df.columns, "The given DataFrame is missing the column given for " \
                                                   "'identifier_label'."
        self.identifier_label: Optional[str] = identifier_label
        self.identifier_value: Optional[Value] = None
        self.values_conditions_identifier = []

        # pleas note that `ValueFactory` uses some fields defined above
        ValueFactory.__init__(self)

        for name in df.columns:
            # we are skipping aliases
            if name in self.column_aliases:
                continue
            if name in self.type_overrides:
                value = self._apply_type_overrides(df, name)
            else:
                identified_value = self.identify_value(col=df[name], name=name)
                # None means the value has already been detected:
                if identified_value is None:
                    continue
                value = identified_value
            if name in self.condition_columns:
                self.conditions.append(value)
            elif name == identifier_label:
                self.identifier_value = value
            else:
                self.values.append(value)
            self.values_conditions_identifier.append(value)

        # Automatic extraction of specification parameters
        df = df.copy()
        for value in self.values_conditions_identifier:
            value.extract(df=df)

        # VAE
        self.vae_series = self.add_module(
            module='vae_series', name='vae', values=self.values, conditions=self.conditions,
            identifier_label=self.identifier_label, identifier_value=self.identifier_value, lstm_mode=self.lstm_mode,
            latent_size=latent_size, network=network, capacity=capacity,
            num_layers=num_layers, residual_depths=residual_depths, batchnorm=batchnorm, activation=activation,
            optimizer=optimizer, learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
            initial_boost=initial_boost, clip_gradients=clip_gradients, beta=beta,
            weight_decay=weight_decay, summarize=(summarizer_dir is not None)
        )

        # Input argument placeholder for num_rows
        self.num_rows: Optional[tf.Tensor] = None

        # Output DF
        self.synthesized: Optional[Dict[str, tf.Tensor]] = None

    def _apply_type_overrides(self, df, name) -> Value:
        assert name in self.type_overrides
        forced_type = self.type_overrides[name]
        if forced_type == TypeOverride.ID:
            value: Value = self.create_identifier(name)
            self.identifier_value = value
        elif forced_type == TypeOverride.CATEGORICAL:
            value = self.create_categorical(name)
        elif forced_type == TypeOverride.CONTINUOUS:
            value = self.create_continuous(name)
        elif forced_type == TypeOverride.DATE:
            value = self.create_date(name)
        elif forced_type == TypeOverride.ENUMERATION:
            value = self.create_enumeration(name)
        else:
            assert False
        is_nan = df[name].isna().any()
        if is_nan and isinstance(value, ContinuousValue):
            value = self.create_nan(name, value)
        return value

    def specification(self):
        spec = super().specification()
        spec.update(
            values=[value.specification() for value in self.values],
            conditions=[value.specification() for value in self.conditions],
            vae=self.vae_series.specification(), batch_size=self.batch_size
        )
        return spec

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()
        for value in self.values:
            df = value.preprocess(df=df)
        return df

    def module_initialize(self):
        super().module_initialize()

        # API function learn

        # Input values
        xs = OrderedDict()
        for value in self.values_conditions_identifier:
            xs.update(zip(value.learned_input_columns(), value.input_tensors()))

        # VAE learn
        self.losses, self.optimized = self.vae_series.learn(xs=xs)

        # VAE learn
        self.learned_seq = self.vae_series.learn(xs=xs, return_sequence=True)

        # # Increment global step
        with tf.control_dependencies(control_inputs=[self.optimized]):
            self.optimized = self.global_step.assign_add(delta=1)

        # Input argument placeholder for num_rows
        self.num_rows = tf.compat.v1.placeholder(dtype=tf.int64, shape=(), name='num_rows')

        # Input condition values
        cs = OrderedDict()
        for value in self.conditions:
            cs.update(zip(value.learned_input_columns(), value.input_tensors()))

        # VAE synthesize
        self.synthesized = self.vae_series.synthesize(n=self.num_rows, cs=cs)

    def learn(
        self, num_iterations: int, df: pd.DataFrame,
        callback: Callable[[Synthesizer, int, dict], bool] = Synthesizer.logging,
        callback_freq: int = 0, verbose=50, print_data=0
    ) -> None:

        assert num_iterations is not None and num_iterations > 0

        df = df.copy()
        for value in self.values_conditions_identifier:
            df = value.preprocess(df=df)

        if self.identifier_label is None:
            num_data = [len(df)]
            groups = [{
                value.name: df[value.name].to_numpy() for value in self.values_conditions_identifier
            }]

        else:
            groups = [group[1] for group in df.groupby(by=self.identifier_label)]
            num_data = [len(group) for group in groups]
            for n, group in enumerate(groups):
                groups[n] = {
                    value.name: df[value.name].to_numpy() for value in self.values_conditions_identifier
                }

        fetches = self.optimized
        # fetches = self.synthesized

        # Batch iteration
        for iteration in range(num_iterations):

            group = randrange(len(num_data))
            data_dict = groups[group]

            data = {
                placeholder: data_dict[name] for value in self.values_conditions_identifier
                for name, placeholder in zip(value.learned_input_columns(), value.input_tensors())
            }

            if num_data[group] > self.max_seq_len:
                start = randrange(num_data[group] - self.max_seq_len)
                batch = np.arange(start, start + self.max_seq_len)
            else:
                batch = np.arange(num_data[group])

            feed_dict = {placeholder: value_data[batch] for placeholder, value_data in data.items()}
            self.run(fetches=fetches, feed_dict=feed_dict)

            if verbose > 0 and (
                iteration == 0 or iteration % verbose == 0
            ):
                group = randrange(len(num_data))
                df = groups[group]
                start = randrange(max(num_data[group] - 1024, 1))
                batch = np.arange(start, min(start + 1024, num_data[group]))

                feed_dict = {placeholder: value_data[batch] for placeholder, value_data in data.items()}
                fetched = self.run(fetches=self.losses, feed_dict=feed_dict)
                self.print_learn_stats(df, batch, fetched, iteration, print_data)

        feed_dict = {placeholder: value_data[batch] for placeholder, value_data in data.items()}
        synth = self.run(fetches=self.learned_seq, feed_dict=feed_dict)

        return [value_data[batch] for value_data in data.values()], synth

    def print_learn_stats(self, df, batch, fetched, iteration, print_data):
        print('ITERATION {iteration} :: Loss: total={loss:.4f} ({losses})'.format(
            iteration=(iteration + 1), loss=fetched['total-loss'], losses=', '.join(
                '{name}={loss:.4f}'.format(name=name, loss=loss)
                for name, loss in fetched.items()
            )
        ))
        # self.loss_history.append(fetched)
        #
        # synthesized = self.synthesize(1000)
        # synthesized = self.preprocess(df=synthesized)
        # dist_by_col = [(col, ks_2samp(df[col], synthesized[col].get_values())[0]) for col in df.keys()]
        # avg_dist = np.mean([dist for (col, dist) in dist_by_col])
        # dists = ', '.join(['{col}={dist:.2f}'.format(col=col, dist=dist) for (col, dist) in dist_by_col])
        # print('KS distances: avg={avg_dist:.2f} ({dists})'.format(avg_dist=avg_dist, dists=dists))
        # self.ks_distance_history.append(dict(dist_by_col))
        #
        # if print_data > 0:
        #     print('original data:')
        #     print(pd.DataFrame.from_dict({key: value[batch] for key, value in df.items()}).head(print_data))
        #     print('synthesized data:')
        #     print(pd.DataFrame.from_dict(fetched['synthesized']).head(print_data))

    def get_loss_history(self):
        return pd.DataFrame.from_records(self.loss_history)

    def get_ks_distance_history(self):
        return pd.DataFrame.from_records(self.ks_distance_history)

    def synthesize(self, num_series: int = None, series_length: int = None, series_lengths: List[int] = None,
                   conditions: Union[dict, pd.DataFrame] = None) -> pd.DataFrame:
        """Synthesize a dataset from the learned model

        :param num_series: Number of series to synthesize.
        :param series_length: Length of the synthesized series, if all have same length.
        :param series_lengths: List of lenghts of each synthesized series, if they are different
        :param conditions: Conditions.
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

        if conditions is not None:
            if isinstance(conditions, dict):
                df_conditions = pd.DataFrame.from_dict(
                    {name: np.reshape(condition, (-1,)) for name, condition in conditions.items()}
                )
            else:
                df_conditions = conditions.copy()
        else:
            df_conditions = None

        for value in self.conditions:
            df_conditions = value.preprocess(df=df_conditions)

        fetches = self.synthesized
        feed_dict = {self.num_rows: series_length}

        # Add conditions to 'feed_dict'
        for value in self.conditions:
            for name, placeholder in zip(value.learned_input_columns(), value.input_tensors()):
                condition = df_conditions[name].values
                if condition.shape == (1,):
                    feed_dict[placeholder] = np.tile(condition, (series_length,))
                elif condition.shape == (series_length,):
                    feed_dict[placeholder] = condition
                else:
                    raise NotImplementedError

        columns = [value.name for value in self.values_conditions_identifier]
        synthesized = None

        if num_series is not None:
            for i in range(num_series):
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                if self.identifier_label:
                    other[self.identifier_label] = [i for _ in range(series_length)]
                other = pd.DataFrame.from_dict(other)[columns]
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        elif series_lengths is not None:
            for i, series_length in enumerate(series_lengths):
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                if self.identifier_label:
                    other[self.identifier_label] = [i for _ in range(series_length)]
                other = pd.DataFrame.from_dict(other)[columns]
                if synthesized is None:
                    synthesized = other
                else:
                    synthesized = synthesized.append(other, ignore_index=True)

        for value in self.values_conditions_identifier:
            synthesized = value.postprocess(df=synthesized)

        return synthesized
