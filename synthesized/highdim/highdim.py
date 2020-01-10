"""This module implements the BasicSynthesizer class."""
import enum
from collections import OrderedDict
from typing import Callable, List, Union, Dict, Set, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import identify_rules, Value, ValueFactory
from ..common.learning_manager import LearningManager
from ..common.util import ProfilerArgs
from ..common.values import ContinuousValue
from ..synthesizer import Synthesizer


class TypeOverride(enum.Enum):
    ID = 'ID'
    DATE = 'DATE'
    CATEGORICAL = 'CATEGORICAL'
    CONTINUOUS = 'CONTINUOUS'
    ENUMERATION = 'ENUMERATION'


class HighDimSynthesizer(Synthesizer,  ValueFactory):
    """The main synthesizer implementation.

    Synthesizer which can learn from data to produce basic tabular data with independent rows, that
    is, no temporal or otherwise conditional relation between the rows.
    """

    def __init__(
        self, df: pd.DataFrame, summarizer_dir: str = None, summarizer_name: str = None,
        profiler_args: ProfilerArgs = None,
        type_overrides: Dict[str, TypeOverride] = None,
        produce_nans_for: Union[bool, Iterable[str], None] = None,
        column_aliases: Dict[str, str] = None,
        # VAE distribution
        distribution: str = 'normal', latent_size: int = 32,
        # Network
        network: str = 'resnet', capacity: int = 128, num_layers: int = 2,
        residual_depths: Union[None, int, List[int]] = 6,
        batchnorm: bool = True, activation: str = 'relu',
        # Optimizer
        optimizer: str = 'adam', learning_rate: float = 3e-3, decay_steps: int = None, decay_rate: float = None,
        initial_boost: int = 500, clip_gradients: float = 1.0, batch_size: int = 1024,
        # Losses
        beta: float = 1.0, weight_decay: float = 1e-3,
        # Categorical
        categorical_weight: float = 3.5, temperature: float = 1.0, moving_average: bool = True,
        # Continuous
        continuous_weight: float = 5.0,
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
        # Evaluation conditions
        learning_manager: bool = True
    ):
        """Initialize a new BasicSynthesizer instance.

        Args:
            df: Data sample which is representative of the target data to generate. Usually, it is
                fine to just use the training data here. Generally, it should exhibit all relevant
                characteristics, so for instance all values a discrete-value column can take.
            summarizer_dir: Directory for TensorBoard summaries, automatically creates unique subfolder.
            profiler_args: A ProfilerArgs object.
            type_overrides: A dict of type overrides per column.
            produce_nans_for: A list containing the columns for which nans will be synthesized. If None or False, no
                column will generate nulls, if True all columns generate nulls (if it applies).
            distribution: Distribution type: "normal".
            latent_size: Latent size.
            network: Network type: "mlp" or "resnet".
            capacity: Architecture capacity.
            num_layers: Architecture depth.
            residual_depths: The depth(s) of each individual residual layer.
            batchnorm: Whether to use batch normalization.
            activation: Activation function.
            optimizer: Optimizer.
            learning_rate: Learning rate.
            decay_steps: Learning rate decay steps.
            decay_rate: Learning rate decay rate.
            initial_boost: Number of steps for initial x10 learning rate boost.
            clip_gradients: Gradient norm clipping.
            batch_size: Batch size.
            beta: VAE KL-loss beta.
            weight_decay: Weight decay.
            categorical_weight: Coefficient for categorical value losses.
            temperature: Temperature for categorical value distributions.
            moving_average: Whether to use moving average scaling for categorical values.
            continuous_weight: Coefficient for continuous value losses.
            condition_columns: ???.
            title_label: Person title column.
            gender_label: Person gender column.
            name_label: Person combined first and last name column.
            firstname_label: Person first name column.
            lastname_label: Person last name column.
            email_label: Person e-mail address column.
            mobile_number_label: Person mobile number column.
            home_number_label: Person home number column.
            work_number_label Person work number column.
            bic_label: BIC column.
            sort_code_label: Bank sort code column.
            account_label: Bank account column.
            postcode_label: Address postcode column.
            county_label: Address county column.
            city_label: Address city column.
            district_label: Address district column.
            street_label: Address street column.
            house_number_label: Address house number column.
            flat_label: Address flat number column.
            house_name_label: Address house column.
            address_label: Address combined column.
            postcode_regex: Address postcode regular expression.
            identifier_label: Identifier column.
            find_rules: List of rules to check for 'all' finds all rules. See
                synthesized.common.values.PairwiseRuleFactory for more examples.
            learning_manager: Whether to use LearningManager.
        """
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
        self.mobile_number_label = mobile_number_label
        self.home_number_label = home_number_label
        self.work_number_label = work_number_label
        self.bic_label = bic_label
        self.sort_code_label = sort_code_label
        self.account_label = account_label
        # Address
        self.address_value: Optional[Value] = None
        self.postcode_label = postcode_label
        self.county_label = county_label
        self.city_label = city_label
        self.district_label = district_label
        self.street_label = street_label
        self.house_number_label = house_number_label
        self.flat_label = flat_label
        self.house_name_label = house_name_label
        self.address_label = address_label
        self.postcode_regex = postcode_regex
        # Identifier
        self.identifier_value: Optional[Value] = None
        self.identifier_label = identifier_label
        # Date
        self.date_value: Optional[Value] = None

        # Values
        self.values: List[Value] = list()
        self.conditions: List[Value] = list()

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
            else:
                self.values.append(value)

        # Automatic extraction of specification parameters
        df = df.copy()
        for value in (self.values + self.conditions):
            value.extract(df=df)

        # Identify deterministic rules
        #  import ipdb; ipdb.set_trace()
        self.values = identify_rules(values=self.values, df=df, tests=self.find_rules)

        # VAE
        self.vae = self.add_module(
            module='vae_old', name='vae', values=self.values, conditions=self.conditions,
            distribution=distribution, latent_size=latent_size, network=network, capacity=capacity,
            num_layers=num_layers, residual_depths=residual_depths, batchnorm=batchnorm, activation=activation,
            optimizer=optimizer, learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
            initial_boost=initial_boost, clip_gradients=clip_gradients, beta=beta,
            weight_decay=weight_decay, summarize=(summarizer_dir is not None)
        )

        # Input argument placeholder for num_rows
        self.num_rows: Optional[tf.Tensor] = None

        # Learning Manager
        self.learning_manager = LearningManager() if learning_manager else None
        self.learning_manager_sample_size = 25_000

    def get_values(self) -> List[Value]:
        return self.values

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
            vae=self.vae.specification(), batch_size=self.batch_size
        )
        return spec

    def preprocess(self, df):
        # TODO: temporary for evaluation notebook!
        df = df.copy()
        for value in (self.values + self.conditions):
            df = value.preprocess(df=df)
        return df

    def module_initialize(self):
        super().module_initialize()

        # API function learn

        # Input values
        xs = OrderedDict()
        for value in (self.values + self.conditions):
            xs.update(zip(value.learned_input_columns(), value.input_tensors()))

        # VAE learn
        self.losses, self.optimized = self.vae.learn(xs=xs)

        # Increment global step
        with tf.control_dependencies(control_inputs=[self.optimized]):
            self.optimized = self.global_step.assign_add(delta=1)

        # Input argument placeholder for num_rows
        self.num_rows = tf.compat.v1.placeholder(dtype=tf.int64, shape=(), name='num_rows')

        # Input condition values
        cs = OrderedDict()
        for value in self.conditions:
            cs.update(zip(value.learned_input_columns(), value.input_tensors()))

        # VAE synthesize
        self.synthesized = self.vae.synthesize(n=self.num_rows, cs=cs)

    def learn(
        self, num_iterations: int, df_train: pd.DataFrame,
        callback: Callable[[Synthesizer, int, dict], bool] = Synthesizer.logging,
        callback_freq: int = 0
    ) -> None:
        """Train the generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            num_iterations: The number of training iterations (not epochs).
            df_train: The training data.
            callback: A callback function, e.g. for logging purposes. Takes the synthesizer
                instance, the iteration number, and a dictionary of values (usually the losses) as
                arguments. Aborts training if the return value is True.
            callback_freq: Callback frequency.

        """
        df_train = df_train.copy()
        df_train_orig = df_train.copy()
        for value in (self.values + self.conditions):
            df_train = value.preprocess(df=df_train)

        num_data = len(df_train)
        data = {
            placeholder: df_train[name].to_numpy() for value in (self.values + self.conditions)
            for name, placeholder in zip(value.learned_input_columns(), value.input_tensors())
        }

        fetches = self.optimized
        callback_fetches = (self.optimized, self.losses)

        for iteration in range(1, num_iterations + 1):
            batch = np.random.randint(num_data, size=self.batch_size)
            feed_dict = {placeholder: value_data[batch] for placeholder, value_data in data.items()}

            if callback is not None and callback_freq > 0 and (
                iteration == 1 or iteration == num_iterations or iteration % callback_freq == 0
            ):
                _, fetched = self.run(fetches=callback_fetches, feed_dict=feed_dict)

                if self.saver is not None and self.summarizer_dir is not None and self.initialized:
                    self.saver.save(sess=self.session, save_path=self.summarizer_dir + 'embeddings.ckpt',
                                    global_step=self.global_step)

                if callback(self, iteration, fetched) is True:
                    return
            else:
                self.run(fetches=fetches, feed_dict=feed_dict)

            if self.learning_manager and self.learning_manager.stop_learning(
                    iteration, synthesizer=self, df_train=df_train_orig, sample_size=self.learning_manager_sample_size
            ):
                break

    def synthesize(
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            progress_callback: Progress bar callback.

        Returns:
            The generated data.

        """

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

        columns = [
            name for value in (self.values + self.conditions)
            for name in value.learned_output_columns()
        ]
        if len(columns) == 0:
            df_synthesized = pd.DataFrame(dict(_sentinel=np.zeros((num_rows,))))

        else:
            fetches = self.synthesized
            feed_dict = {self.num_rows: (num_rows % 1024)}
            for value in self.conditions:
                for name, placeholder in zip(value.learned_input_columns(), value.input_tensors()):
                    condition = df_conditions[name].values
                    if condition.shape == (1,):
                        feed_dict[placeholder] = np.tile(condition, (num_rows % 1024,))
                    elif condition.shape == (num_rows,):
                        feed_dict[placeholder] = condition[-num_rows % 1024:]
                    else:
                        raise NotImplementedError
            synthesized = self.run(fetches=fetches, feed_dict=feed_dict)
            df_synthesized = pd.DataFrame.from_dict(synthesized)[columns]

            feed_dict = {self.num_rows: 1024}
            for value in self.conditions:
                for name, placeholder in zip(value.learned_input_columns(), value.input_tensors()):
                    condition = df_conditions[name].values
                    if condition.shape == (1,):
                        feed_dict[placeholder] = np.tile(condition, (1024,))
            n_batches = num_rows // 1024
            for k in range(n_batches):
                for value in self.conditions:
                    for name, placeholder in zip(value.learned_input_columns(), value.input_tensors()):
                        condition = df_conditions[name].values
                        if condition.shape == (num_rows,):
                            feed_dict[placeholder] = condition[k * 1024: (k + 1) * 1024]
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                df_synthesized = df_synthesized.append(
                    pd.DataFrame.from_dict(other)[columns], ignore_index=True
                )
                if progress_callback is not None:
                    # report approximate progress from 0% to 98% (2% are reserved for post actions)
                    progress_callback(round((k + 1) * 98.0 / n_batches))
        for value in (self.values + self.conditions):
            df_synthesized = value.postprocess(df=df_synthesized)

        if len(columns) == 0:
            df_synthesized.pop('_sentinel')

        # aliases:
        for alias, col in self.column_aliases.items():
            df_synthesized[alias] = df_synthesized[col]

        assert len(df_synthesized.columns) == len(self.columns)
        df_synthesized = df_synthesized[self.columns]

        return df_synthesized
