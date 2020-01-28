"""This module implements the BasicSynthesizer class."""
from typing import Callable, List, Union, Dict, Iterable, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import Value, ValueFactory, TypeOverride
from ..common.learning_manager import LearningManager
from ..common.util import ProfilerArgs, record_summaries_every_n_global_steps
from ..common.generative import VAEOld
from ..common.synthesizer import Synthesizer

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class HighDimSynthesizer(Synthesizer):
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
        initial_boost: int = 0, clip_gradients: float = 1.0,
        # Batch size
        batch_size: int = 64, increase_batch_size_every: Optional[int] = 500, max_batch_size: Optional[int] = 1024,
        # Losses
        beta: float = 1.0, weight_decay: float = 1e-3,
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
        super(HighDimSynthesizer, self).__init__(
            name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name
        )
        self.value_factory = ValueFactory(
            name='value_factory', df=df,
            capacity=capacity, weight_decay=weight_decay,
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
        self.batch_size = batch_size
        self.tf_batch_size = tf.Variable(initial_value=batch_size, dtype=tf.int64)
        self.increase_batch_size_every = increase_batch_size_every
        self.max_batch_size: int = max_batch_size if max_batch_size else batch_size

        # VAE
        self.vae = VAEOld(
            name='vae', values=self.get_values(), conditions=self.get_conditions(),
            distribution=distribution, latent_size=latent_size, network=network, capacity=capacity,
            num_layers=num_layers, residual_depths=residual_depths, batchnorm=batchnorm, activation=activation,
            optimizer=optimizer, learning_rate=tf.constant(learning_rate, dtype=tf.float32), decay_steps=decay_steps,
            decay_rate=decay_rate, initial_boost=initial_boost, clip_gradients=clip_gradients, beta=beta,
            weight_decay=weight_decay, summarize=(summarizer_dir is not None)
        )

        # Input argument placeholder for num_rows
        self.num_rows: Optional[tf.Tensor] = None

        # Learning Manager
        self.learning_manager: Optional[LearningManager] = None
        if learning_manager:
            self.use_vae_loss = True
            self.learning_manager = LearningManager()
            self.learning_manager.set_check_frequency(self.batch_size)

    def get_values(self) -> List[Value]:
        return self.value_factory.get_values()

    def get_conditions(self) -> List[Value]:
        return self.value_factory.get_conditions()

    def get_losses(self) -> tf.Tensor:
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
        callback_freq: int = 0
    ) -> None:
        """Train the generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            df_train: The training data.
            num_iterations: The number of training iterations (not epochs).
            callback: A callback function, e.g. for logging purposes. Takes the synthesizer
                instance, the iteration number, and a dictionary of values (usually the losses) as
                arguments. Aborts training if the return value is True.
            callback_freq: Callback frequency.

        """
        assert num_iterations or self.learning_manager, "'num_iterations' must be set if learning_manager=False"

        df_train_orig = df_train
        df_train = self.value_factory.preprocess(df_train)
        data = self.value_factory.get_data_feed_dict(df_train)

        num_data = len(df_train)

        with record_summaries_every_n_global_steps(callback_freq, self.global_step):
            keep_learning = True
            iteration = 0
            while keep_learning:
                batch = tf.random.uniform(shape=(self.batch_size,), maxval=len(df_train), dtype=tf.int64)
                feed_dict = {name: tf.nn.embedding_lookup(params=value_data, ids=batch)
                             for name, value_data in data.items()}

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

                    # if callback(self, iteration, fetched) is True:
                    #     return
                else:
                    self.vae.learn(xs=feed_dict)

                if self.learning_manager:
                    if self.learning_manager.stop_learning(iteration, synthesizer=self, use_vae_loss=self.use_vae_loss,
                                                           data_dict=data, num_data=num_data,
                                                           df_train_orig=df_train_orig):
                        break

                # Increase batch size
                tf.summary.scalar(name='batch_size', data=self.tf_batch_size)
                if self.increase_batch_size_every and iteration > 0 and self.batch_size < self.max_batch_size and \
                        iteration % self.increase_batch_size_every == 0:
                    self.batch_size *= 2
                    if self.batch_size > self.max_batch_size:
                        self.batch_size = self.max_batch_size
                    if self.batch_size == self.max_batch_size:
                        logger.info('Maximum batch size of {} reached.'.format(self.max_batch_size))
                    if self.learning_manager:
                        self.learning_manager.set_check_frequency(self.batch_size)
                    self.tf_batch_size.assign(self.batch_size)
                    logger.info('Iteration {} :: Batch size increased to {}'.format(iteration, self.batch_size))

                # Increment iteration number, and check if we reached max num_iterations
                iteration += 1
                if num_iterations:
                    keep_learning = iteration < num_iterations

                self.global_step.assign_add(1)

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

        df_conditions = self.value_factory.preprocess_conditions(conditions)
        columns = self.value_factory.get_column_names()

        if len(columns) == 0:
            return pd.DataFrame([[], ]*num_rows)

        feed_dict = self.value_factory.get_conditions_feed_dict(df_conditions, num_rows)
        synthesized = self.vae.synthesize(tf.constant(num_rows % 1024, dtype=tf.int64), cs=feed_dict)
        df_synthesized = pd.DataFrame.from_dict(synthesized)[columns]

        feed_dict = self.value_factory.get_conditions_feed_dict(df_conditions, 1024)
        n_batches = num_rows // 1024
        data = self.value_factory.get_conditions_data(df_conditions)

        if self.writer is not None:
            tf.summary.trace_on(graph=True, profiler=False)

        for k in range(n_batches):
            feed_dict.update({
                placeholder: tf.constant(condition_data[k * 1024: (k + 1) * 1024], dtype=tf.float32)
                for placeholder, condition_data in data.items()
                if condition_data.shape == (num_rows,)
            })
            other = self.vae.synthesize(tf.constant(1024, dtype=tf.int64), cs=feed_dict)
            df_synthesized = df_synthesized.append(
                pd.DataFrame.from_dict(other)[columns], ignore_index=True
            )
            if progress_callback is not None:
                # report approximate progress from 0% to 98% (2% are reserved for post actions)
                progress_callback(round((k + 1) * 98.0 / n_batches))

        df_synthesized = self.value_factory.postprocess(df_synthesized)

        if self.writer is not None:
            tf.summary.trace_export(name='Synthesize', step=0)
            tf.summary.trace_off()

        return df_synthesized

    def encode(self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encodes dataset and returns the corresponding latent space and generated data.

        Args:
            df_encode: Input dataset
            conditions: The condition values for the generated rows.

        Returns:
            (Pandas DataFrame of latent space, Pandas DataFrame of decoded space) corresponding to input data
        """
        df_encode = df_encode.copy()
        for value in (self.values + self.conditions):
            df_encode = value.preprocess(df=df_encode)

        num_rows = len(df_encode)
        feed_dict = {
            name: df_encode[name].to_numpy() for value in (self.values + self.conditions)
            for name in value.learned_input_columns()
        }
        feed_dict[self.num_rows] = num_rows

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
            for name in value.learned_input_columns():
                condition = df_conditions[name].values
                if condition.shape == (1,):
                    feed_dict[name] = np.tile(condition, (num_rows,))
                elif condition.shape == (num_rows,):
                    feed_dict[name] = condition[-num_rows:]
                else:
                    raise NotImplementedError

        encoded, decoded = self.run(fetches=(self.xs_latent_space, self.xs_synthesized), feed_dict=feed_dict)

        columns = [
            name for value in (self.values + self.conditions)
            for name in value.learned_output_columns()
        ]
        df_synthesized = pd.DataFrame.from_dict(decoded)[columns]

        for value in (self.values + self.conditions):
            df_synthesized = value.postprocess(df=df_synthesized)

        # aliases:
        for alias, col in self.column_aliases.items():
            df_synthesized[alias] = df_synthesized[col]

        assert len(df_synthesized.columns) == len(self.columns)
        df_synthesized = df_synthesized[self.columns]

        latent = np.concatenate((encoded['sample'], encoded['mean'], encoded['std']), axis=1)
        df_encoded = pd.DataFrame.from_records(latent, columns=[f"{l}_{n}" for l in 'lms'
                                                                for n in range(encoded['sample'].shape[1])])

        return df_encoded, df_synthesized
