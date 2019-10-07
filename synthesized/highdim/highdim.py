"""This module implements the BasicSynthesizer class."""
import enum
from collections import OrderedDict
from typing import Callable, List, Union, Dict, Set, Iterable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import identify_rules, Value, ValueFactory
from ..common.values import ContinuousValue
from ..common.util import ProfilerArgs
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
        self, df: pd.DataFrame, summarizer_dir: str = None, profiler_args: ProfilerArgs = None,
        type_overrides: Dict[str, TypeOverride] = None,
        produce_nans_for: Iterable[str] = None,
        # VAE distribution
        distribution: str = 'normal', latent_size: int = 128,
        # Network
        network: str = 'resnet', capacity: int = 128, depth: int = 2, batchnorm: bool = True,
        activation: str = 'relu',
        # Optimizer
        optimizer: str = 'adam', learning_rate: float = 3e-4, decay_steps: int = 1000,
        decay_rate: float = 0.9, initial_boost: bool = False, clip_gradients: float = 1.0,
        batch_size: int = 64,
        # Losses
        categorical_weight: float = 1.0, continuous_weight: float = 1.0, beta: float = 0.064,
        weight_decay: float = 1e-5,
        # Categorical
        temperature: float = 1.0, smoothing: float = 0.1, moving_average: bool = True,
        similarity_regularization: float = 0.1, entropy_regularization: float = 0.1,
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
        find_rules: Union[str, List[str]] = None
    ):
        """Initialize a new BasicSynthesizer instance.

        Args:
            df: Data sample which is representative of the target data to generate. Usually, it is
                fine to just use the training data here. Generally, it should exhibit all relevant
                characteristics, so for instance all values a discrete-value column can take.
            summarizer: Directory for TensorBoard summaries, automatically creates unique subfolder.
            profiler_args: A ProfilerArgs object.
            type_overrides: A dict of type overrides per column.
            distribution: Distribution type: "normal".
            latent_size: Latent size.
            network: Network type: "mlp" or "resnet".
            capacity: Architecture capacity.
            depth: Architecture depth.
            batchnorm: Whether to use batch normalization.
            activation: Activation function.
            optimizer: Optimizer.
            learning_rate: Learning rate.
            decay_steps: Learning rate decay steps.
            decay_rate: Learning rate decay rate.
            initial_boost: Learning rate boost for initial steps.
            clip_gradients: Gradient norm clipping.
            batch_size: Batch size.
            categorical_weight: Coefficient for categorical value losses.
            continuous_weight: Coefficient for continuous value losses.
            beta: VAE KL-loss beta.
            weight_decay: Weight decay.
            temperature: Temperature for categorical value distributions.
            smoothing: Smoothing for categorical value distributions.
            moving_average: Whether to use moving average scaling for categorical values.
            similarity_regularization: Similarity regularization coefficient for categorical values.
            entropy_regularization: Entropy regularization coefficient for categorical values.
            condition_columns: ???.
            title_label: Person title column.
            gender_label: Person gender column.
            name_label: Person combined first and last name column.
            firstname_label: Person first name column.
            lastname_label: Person last name column.
            email_label: Person e-mail address column.
            postcode_label: Address postcode column.
            city_label: Address city column.
            street_label: Address street column.
            address_label: Address combined column.
            postcode_regex: Address postcode regular expression.
            identifier_label: Identifier column.
            find_rules: List of rules to check for 'all' finds all rules. See
                synthesized.common.values.PairwiseRuleFactory for more examples.
        """
        Synthesizer.__init__(self, name='synthesizer', summarizer_dir=summarizer_dir, profiler_args=profiler_args)
        if type_overrides is None:
            self.type_overrides: Dict[str, TypeOverride] = dict()
        else:
            self.type_overrides = type_overrides

        if produce_nans_for is None:
            self.produce_nans_for: Set[str] = set()
        else:
            self.produce_nans_for = set(produce_nans_for)

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
        self.categorical_weight = categorical_weight
        self.continuous_weight = continuous_weight
        self.temperature = temperature
        self.smoothing = smoothing
        self.moving_average = moving_average
        self.similarity_regularization = similarity_regularization
        self.entropy_regularization = entropy_regularization

        # Overall columns
        self.columns = list(df.columns)
        # Person
        self.person_value: Optional[Value] = None
        self.title_label = title_label
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        # Address
        self.address_value: Optional[Value] = None
        self.postcode_label = postcode_label
        self.city_label = city_label
        self.street_label = street_label
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
            if name in self.type_overrides:
                value = self._apply_type_overrides(df, name)
            else:
                value = self.identify_value(col=df[name], name=name)
            assert len(value.columns()) == 1 and value.columns()[0] == name
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
            depth=depth, batchnorm=batchnorm, activation=activation, optimizer=optimizer,
            learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
            initial_boost=initial_boost, clip_gradients=clip_gradients, beta=beta,
            weight_decay=weight_decay, summarize=(summarizer_dir is not None)
        )

        # Input argument placeholder for num_rows
        self.num_rows: Optional[tf.Tensor] = None

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

        # API function synthesize

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
                if callback(self, iteration, fetched) is True:
                    return
            else:
                self.run(fetches=fetches, feed_dict=feed_dict)

    def synthesize(
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.

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
            for k in range(num_rows // 1024):
                for value in self.conditions:
                    for name, placeholder in zip(value.learned_input_columns(), value.input_tensors()):
                        condition = df_conditions[name].values
                        if condition.shape == (num_rows,):
                            feed_dict[placeholder] = condition[k * 1024: (k + 1) * 1024]
                other = self.run(fetches=fetches, feed_dict=feed_dict)
                df_synthesized = df_synthesized.append(
                    pd.DataFrame.from_dict(other)[columns], ignore_index=True
                )

        for value in (self.values + self.conditions):
            df_synthesized = value.postprocess(df=df_synthesized)

        if len(columns) == 0:
            df_synthesized.pop('_sentinel')

        assert len(df_synthesized.columns) == len(self.columns)
        df_synthesized = df_synthesized[self.columns]

        return df_synthesized
