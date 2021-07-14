"""This module implements the BasicSynthesizer class."""
import logging
import pickle
from dataclasses import asdict
from math import sqrt
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .binary_builder import ModelBinary
from ..common.generative import HighDimEngine
from ..common.learning_manager import LearningManager
from ..common.rules import Association, Expression, GenericRule
from ..common.synthesizer import Synthesizer
from ..common.util import get_privacy_budget, record_summaries_every_n_global_steps
from ..common.values import DataFrameValue, ValueExtractor
from ..config import EngineConfig, HighDimConfig
from ..metadata import DataFrameMeta
from ..model import ContinuousModel, DataFrameModel, DiscreteModel, Model
from ..model.factory import ModelFactory
from ..model.models import AddressModel, BankModel, EnumerationModel, FormattedStringModel, GenderModel, PersonModel
from ..transformer import DataFrameTransformer
from ..version import __version__
from ..licence import LicenceError, OptionalFeature, verify

logger = logging.getLogger(__name__)


class HighDimSynthesizer(Synthesizer):
    """Synthesizer that can learn to generate data from a single tabular dataset.

    The data must be in a tabular format (i.e a set of columns and rows), where each row is independent and there
    is no temporal or conditional relation between them. The Synthesizer will learn the underlying distribution
    of the original data, and is capable of generating new synthetic rows of data capture maintain the correlations
    and associations across columns.

    Args:
        df_meta (DataFrameMeta): A :class:`~synthesized.DataFrameMeta` instance that has been extracted for the
            desired dataset.
        config (HighDimConfig, optional): The configuration to use for this Synthesizer. Defaults to None, in which
            case the default options of :class:`~synthesized.config.HighDimConfig` are used.
        type_overrides (List[Union[ContinuousModel, DiscreteModel]], Optional): Custom type specifciations for each
            column that will override the defaults inferred from the data. These must be instantiated Model classes, e.g
            :class:`~synthesized.model.models.Histogram` or :class:`~synthesized.model.models.KernelDensityEstimate`.
            Defaults to None, in which case the types are automatically inferred.
        summarizer_dir (str, optional): Path to a directory where TensorBoard summaries of the training logs
            will be stored. Defaults to None.
        summarizer_name (str, optional): A prefix for the subdirectory where trainining logs for this Synthesizer
            are stored. If set, logs will be stored in ``summarizer_dir/summarizer_name_%Y%m%d-%H%M%S``, where
            the timestamp is set at the time of instantiation. Defaults to None.

    Examples:

        Load dataset into a pandas DataFrame:

        >>> df = pd.read_csv('dataset.csv')

        Extract the DataFrameMeta:

        >>> df_meta = MetaExtractor.extract(df)

        Initialise a ``HighDimSynthesizer`` with the default configuration:

        >>> synthesizer = HighDimSynthesizer(df_meta=df_meta)

        Learn a model of the original data by training for 100 iterations:

        >>> synthesizer.learn(df_train=df, num_iterations=100)

        Generate 1000 rows of new data:

        >>> df_synthetic = synthesizer.synthesize(num_rows=1000)

        Set a column to be categorical instead of continuous:

        >>> column_meta = df_meta['column_name']
        >>> column_model = synthesized.model.models.Histogram(meta=column_meta)
        >>> synthesizer = HighDimSynthesizer(df_meta=df_meta, type_overrides=[column_model])
    """

    def __init__(
            self, df_meta: DataFrameMeta, config: HighDimConfig = None,
            type_overrides: List[Union[ContinuousModel, DiscreteModel]] = None, summarizer_dir: str = None,
            summarizer_name: str = None,
    ):
        super(HighDimSynthesizer, self).__init__(
            name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name
        )
        config = config or HighDimConfig()
        self.config = config
        self.type_overrides = type_overrides
        if config.engine_config.differential_privacy:
            if verify(OptionalFeature.DIFFERENTIAL_PRIVACY):
                self._differential_privacy = True
                self._privacy_config = config.privacy_config
                self._epsilon: Optional[float] = None
                config.increase_batch_size_every = None  # privacy accounting requires a constant batch size, so keep fixed.
            else:
                raise LicenceError('Please upgrade your licence to use differential privacy features.')
        else:
            self._differential_privacy = False

        self._init_engine(df_meta, type_overrides)

        # Learning Manager
        self._learning_manager: Optional[LearningManager] = None
        if config.learning_manager:
            use_engine_loss = False if config.custom_stop_metric else True
            self._learning_manager = LearningManager(
                max_training_time=config.max_training_time, use_engine_loss=use_engine_loss,
                custom_stop_metric=config.custom_stop_metric, sample_size=1024
            )

    def __repr__(self):
        return f"HighDimSynthesizer(df_meta={self.df_meta}, type_overrides={self.type_overrides})"

    @property
    def epsilon(self) -> Optional[float]:
        """Value of epsilon obtained by this Synthesizer if differential privacy is enabled."""
        if self._differential_privacy:
            return self._epsilon
        else:
            return None

    def _init_engine(
            self, df_meta: DataFrameMeta,
            type_overrides: Optional[List[Union[ContinuousModel, DiscreteModel]]] = None
    ):
        model_factory = ModelFactory(config=self.config.model_builder_config)
        self.df_meta = df_meta
        self._df_model: DataFrameModel = model_factory(df_meta, type_overrides=type_overrides)
        self._df_model_independent = self._split_df_model(self._df_model)

        self._df_value: DataFrameValue = ValueExtractor.extract(
            df_meta=self._df_model, name='data_frame_value', config=self.config.value_factory_config
        )

        self._df_transformer: DataFrameTransformer = DataFrameTransformer.from_meta(self._df_model)
        self._engine = HighDimEngine(name='vae', df_value=self._df_value, config=self.config.engine_config)

    def _get_data_feed_dict(self, df: pd.DataFrame) -> Dict[str, Sequence[tf.Tensor]]:
        data: Dict[str, Sequence[tf.Tensor]] = {
            name: tuple([
                tf.constant(df[column])
                for column in value.columns()
            ])
            for name, value in self._df_value.items()
        }
        return data

    def _get_losses(self, data: Dict[str, Sequence[tf.Tensor]] = None) -> Dict[str, tf.Tensor]:
        if data is not None:
            self._engine.loss(data)

        losses = {
            'total-loss': self._engine.total_loss,
            'kl-loss': self._engine.kl_loss,
            'regularization-loss': self._engine.regularization_loss,
            'reconstruction-loss': self._engine.reconstruction_loss
        }
        return losses

    @staticmethod
    def _split_df_model(df_model: DataFrameModel) -> DataFrameModel:
        """Given a df_model, pop out those models that are not learned in the engine and
        return a df_model_independent containing these models.
        """
        df_meta_independent = DataFrameMeta(name='independent_models')
        df_model_independent = DataFrameModel(df_meta_independent)
        models_to_pop: List[str] = []
        models_to_add: List[Model] = []
        for name, model in df_model.items():
            if isinstance(model, EnumerationModel):
                models_to_pop.append(name)

            if isinstance(model, ContinuousModel):
                continue

            elif isinstance(model, (BankModel, FormattedStringModel)):
                models_to_pop.append(name)

            elif isinstance(model, PersonModel) and isinstance(model.hidden_model, GenderModel):
                models_to_add.append(model.hidden_model)
                models_to_pop.append(name)

            elif isinstance(model, AddressModel):
                models_to_add.append(model.postcode_model)
                models_to_pop.append(name)

            elif isinstance(model, DiscreteModel):
                assert model.num_rows
                if len(model.categories) > sqrt(model.num_rows):
                    models_to_pop.append(name)
        annotations = [name for name in models_to_pop if name in df_model.meta.annotations]

        for model_name in models_to_pop:
            df_model_independent[model_name] = df_model.pop(model_name)
        df_model_independent.meta.annotations = annotations

        for model in models_to_add:
            df_model[model.name] = model

        return df_model_independent

    def learn(
            self, df_train: pd.DataFrame, num_iterations: Optional[int] = None,
            callback: Callable[[Synthesizer, int, dict], bool] = None,
            callback_freq: int = 0
    ) -> None:
        """Learn the underlying distribution of the original data by training the synthesizer.

        This method can be called multiple times to continue the training process.

        Args:
            df_train (pd.DataFrame): Training data that matches schema of the DataFrameMeta used by this synthesizer.
            num_iterations (int, optional) The number of training iterations (not epochs). Defaults to None,
                in which case the learning process is intelligently stopped as the synthesizer converges.
            callback (Callable, optional) A callback function that can be used for logging purposes. This takes the
                synthesizer instance, the iteration number, and a dictionary of the loss function values as
                arguments. Aborts training if the return value is True. Defaults to None.
            callback_freq (int, optional): The number of training iterations to perform before the callback is called.
                Defaults to 0, in which case the callback is never called.
        """
        assert num_iterations or self._learning_manager, "'num_iterations' must be set if learning_manager=False"
        self._engine.build()
        if self._learning_manager:
            self._learning_manager.restart_learning_manager()
            self._learning_manager.set_check_frequency(self.config.batch_size)

        if not self.df_meta._extracted:
            self.df_meta.extract(df_train)

        self._df_model_independent.fit(df=df_train)

        if not self._df_transformer.is_fitted():
            self._df_transformer.fit(df_train)

        df_train_pre = self._df_transformer.transform(df_train)
        if self._df_value.learned_input_size() == 0 and len(self._df_model) == 0:
            return

        num_data = len(df_train)
        max_batch_size = self.config.max_batch_size if self.config.max_batch_size else self.config.batch_size

        with self._writer.as_default():
            with record_summaries_every_n_global_steps(callback_freq, self._global_step):
                keep_learning = True
                iteration = 0
                while keep_learning:
                    # Increment iteration number, and check if we reached max num_iterations
                    self._global_step.assign_add(1)
                    iteration += 1
                    if num_iterations:
                        keep_learning = iteration <= num_iterations
                    if keep_learning is False:
                        break

                    batch = tf.random.uniform(shape=(self.config.batch_size,), maxval=num_data, dtype=tf.int64)
                    feed_dict = self._get_data_feed_dict(df_train_pre.iloc[batch])

                    self._engine.learn(xs=feed_dict)

                    if callback is not None and callback_freq > 0 and (
                        iteration == 1 or iteration == num_iterations or iteration % callback_freq == 0
                    ):
                        losses = self._get_losses()
                        if callback(self, iteration, losses) is True:
                            return

                    if self._learning_manager and self._learning_manager.stop_learning(iteration, synthesizer=self):
                        break

                    if self._differential_privacy:
                        self._epsilon = get_privacy_budget(
                            self._privacy_config.noise_multiplier, iteration, self.config.batch_size,
                            num_data, self._privacy_config.delta
                        )
                        if self.epsilon > self._privacy_config.epsilon:
                            break

                    # Increase batch size
                    tf.summary.scalar(name='batch_size', data=self.config.batch_size)
                    if self.config.increase_batch_size_every and iteration > 0 and self.config.batch_size < max_batch_size \
                            and iteration % self.config.increase_batch_size_every == 0:
                        self.config.batch_size *= 2
                        if self.config.batch_size > max_batch_size:
                            self.config.batch_size = max_batch_size

                        if self.config.batch_size == max_batch_size:
                            logger.info('Maximum batch size of {} reached.'.format(max_batch_size))
                        if self._learning_manager:
                            self._learning_manager.set_check_frequency(self.config.batch_size)

    def synthesize_from_rules(
            self, num_rows: int, produce_nans: bool = False, generic_rules: List[GenericRule] = None,
            association_rules: List[Association] = None, expression_rules: List[Expression] = None, max_iter: int = 20,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        """Generate a given number of data rows according to specified rules.

        Conditional sampling is used to generate a dataset that conforms to the the given generic_rules. As a result,
        in some cases it may not be possible to generate num_rows of synthetic data if the original data contains a
        small number of samples where the rule is valid. Increasing max_iter may help in this situation.

        Args:
            num_rows (int): The number of rows to generate.
            produce_nans (bool, optional): Whether to produce NaNs. Defaults to False
            generic_rules (List[GenericRule], optional): list of GenericRule rules the output must conform to.
                Defaults to None.
            association_rules (List[Association], optional): list of Association rules to constrain the output data.
                Defaults to None.
            expression_rules (List[Expression], optional): list of Expression rules to add to the output of the
                synthesizer. Defaults to None.
            max_iter (int, optional): maximum number of iterations to try to apply generic rules before raising an
                error. Defaults to 20.
            progress_callback (Callable, optional): Progress bar callback. Defaults to None.

        Returns:
            The generated data.

        Raises:
            RuntimeError: if num_rows of data can't be generated within max_iter iterations.
        """
        association_rules = association_rules or []
        generic_rules = generic_rules or []
        expression_rules = expression_rules or []

        df = pd.DataFrame({})
        n_missing = num_rows
        for i in range(max_iter):
            df_ = self.synthesize(n_missing, association_rules=association_rules,
                                  produce_nans=produce_nans, progress_callback=progress_callback)
            for generic_rule in generic_rules:
                df_ = generic_rule.filter(df_)
            df = pd.concat((df, df_), ignore_index=True)
            n_missing = num_rows - len(df)
            if not n_missing:
                break
            if i + 1 == max_iter:
                raise RuntimeError(f"HighDimSynthesizer has tried max_iter: {max_iter} number of times to generate "
                                   "data constrained to the generic rules but failed. Try again with a higher max_iter "
                                   "value or less strict generic rules")

        for expression_rule in expression_rules:
            df = expression_rule.apply(df)

        return df

    def synthesize(
            self, num_rows: int, produce_nans: bool = False,
            progress_callback: Callable[[int], None] = None,
            association_rules: List[Association] = None,
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows (int): Number of rows to generate.
            produce_nans (bool, optional): Generate NaN values. Defaults to False.
            progress_callback (Callable, optional): Progress bar callback. Defaults to None.
            association_rules (List[Association], optional): Association rules to apply. Defaults to None.

        Returns:
            The generated data.
        """
        if progress_callback is not None:
            progress_callback(0)

        if num_rows <= 0:
            raise ValueError("Given 'num_rows' must be greater than zero, given '{}'.".format(num_rows))

        if association_rules is not None:
            Association._validate_association_rules(association_rules)

        columns = self.df_meta.columns

        assert columns is not None
        if len(columns) == 0:
            return pd.DataFrame([[], ] * num_rows)

        if self._df_value.learned_output_size() > 0:
            if self.config.synthesis_batch_size is None or self.config.synthesis_batch_size > num_rows:
                synthesized = self._engine.synthesize(
                    tf.constant(num_rows, dtype=tf.int64), association_rules=association_rules)
                assert synthesized is not None
                synthesized = self._df_value.split_outputs(synthesized)
                df_synthesized = pd.DataFrame.from_dict(synthesized)
                if progress_callback is not None:
                    progress_callback(98)

            else:
                dict_synthesized = None
                if num_rows % self.config.synthesis_batch_size > 0:
                    synthesized = self._engine.synthesize(
                        tf.constant(num_rows % self.config.synthesis_batch_size, dtype=tf.int64),
                        association_rules=association_rules
                    )
                    assert synthesized is not None
                    dict_synthesized = self._df_value.split_outputs(synthesized)
                    dict_synthesized = {k: v.tolist() for k, v in dict_synthesized.items()}

                n_batches = num_rows // self.config.synthesis_batch_size

                for k in range(n_batches):
                    other = self._engine.synthesize(tf.constant(self.config.synthesis_batch_size, dtype=tf.int64),
                                                    association_rules=association_rules)
                    other = self._df_value.split_outputs(other)
                    if dict_synthesized is None:
                        dict_synthesized = other
                        dict_synthesized = {key: val.tolist() for key, val in dict_synthesized.items()}
                    else:
                        for c in other.keys():
                            dict_synthesized[c].extend(other[c].tolist())

                    if progress_callback is not None:
                        # report approximate progress from 0% to 98% (2% are reserved for post actions)
                        progress_callback(round((k + 1) * 98.0 / n_batches))

                df_synthesized = pd.DataFrame.from_dict(dict_synthesized)

            df_synthesized = self._df_transformer.inverse_transform(df_synthesized, produce_nans=produce_nans)

        else:
            df_synthesized = pd.DataFrame([[], ] * num_rows)

        df_sampled = self._df_model_independent.sample(n=num_rows, produce_nans=produce_nans, conditions=df_synthesized)
        df_independent = df_sampled[[c for c in df_sampled.columns if c not in df_synthesized.columns]]

        df_synthesized = pd.concat((df_synthesized, df_independent), axis=1)
        df_synthesized = df_synthesized[columns]

        if progress_callback is not None:
            progress_callback(100)

        return df_synthesized

    def _encode(
            self, df_encode: pd.DataFrame, produce_nans: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encodes dataset and returns the corresponding latent space and generated data.

        Args:
            df_encode: Input dataset.
            produce_nans: Whether to produce NaNs.

        Returns:
            (Pandas DataFrame of latent space, Pandas DataFrame of decoded space) corresponding to input data
        """
        df_encode = df_encode.copy()
        df_encode = self._df_transformer.transform(df=df_encode)

        data = self._get_data_feed_dict(df_encode)
        encoded, decoded = self._engine.encode(xs=data)

        decoded = self._df_value.split_outputs(decoded)
        df_synthesized = pd.DataFrame.from_dict(decoded)
        df_synthesized = self._df_transformer.inverse_transform(df_synthesized, produce_nans=produce_nans)

        df_sampled = self._df_model_independent.sample(
            n=len(df_synthesized), produce_nans=produce_nans, conditions=df_synthesized)
        df_independent = df_sampled[[c for c in df_sampled.columns if c not in df_synthesized.columns]]
        df_synthesized = pd.concat((df_synthesized, df_independent), axis=1)
        df_synthesized = df_synthesized[self.df_meta.columns]

        latent = np.concatenate((encoded['sample'], encoded['mean'], encoded['std']), axis=1)
        df_encoded = pd.DataFrame.from_records(latent, columns=[f"{ls}_{n}" for ls in 'lms'
                                                                for n in range(encoded['sample'].shape[1])])

        return df_encoded, df_synthesized

    def _encode_deterministic(
            self, df_encode: pd.DataFrame, produce_nans: bool = False
    ) -> pd.DataFrame:
        """Deterministically encodes a dataset and returns it with imputed nans.

        Args:
            df_encode: Input dataset
            produce_nans: Whether to produce NaNs.

        Returns:
            Pandas DataFrame of decoded space corresponding to input data
        """
        df_orig = df_encode
        df_encode = self._df_transformer.transform(df=df_encode.copy().reset_index(drop=True))
        data = self._get_data_feed_dict(df_encode)
        decoded = self._engine.encode_deterministic(xs=data)
        decoded = self._df_value.split_outputs(decoded)
        df_synthesized = pd.DataFrame.from_dict(decoded)
        df_synthesized = self._df_transformer.inverse_transform(df_synthesized, produce_nans=produce_nans)

        df_independent: pd.DataFrame = df_orig.reset_index(drop=True)[
            [c for c in df_orig.columns if c not in df_synthesized.columns]]
        if not produce_nans:
            df_sampled = self._df_model_independent.sample(
                n=len(df_synthesized), produce_nans=False, conditions=df_synthesized)
            df_sampled = df_sampled[[c for c in df_independent.columns]]
            df_independent = df_independent.where(df_independent.notna(), other=df_sampled)

        df_synthesized = pd.concat((df_synthesized, df_independent), axis=1)

        df_synthesized = df_synthesized[self.df_meta.columns]
        df_synthesized.index = df_orig.index.copy()
        return df_synthesized

    def _get_variables(self) -> Dict[str, Any]:
        variables = super()._get_variables()
        variables.update(
            config=asdict(self.config),
            df_meta=self.df_meta.to_dict(),
            df_model=self._df_model.to_dict(),
            df_model_independent=self._df_model_independent.to_dict(),
            df_value=self._df_value.to_dict(),
            df_transformer=self._df_transformer.to_dict(),

            # VAE
            engine=self._engine.get_variables(),
            latent_size=self._engine.latent_size,
            network=self._engine.network,
            capacity=self._engine.capacity,
            num_layers=self._engine.num_layers,
            residual_depths=self._engine.residual_depths,
            batch_norm=self._engine.batch_norm,
            activation=self._engine.activation,
            optimizer=self._engine.optimizer_name,
            learning_rate=self._engine.learning_rate,
            decay_steps=self._engine.decay_steps,
            decay_rate=self._engine.decay_rate,
            initial_boost=self._engine.initial_boost,
            clip_gradients=self._engine.clip_gradients,
            beta=self._engine.beta,
            weight_decay=self._engine.weight_decay,

            # Learning Manager
            learning_manager=self._learning_manager.get_variables() if self._learning_manager else None
        )

        return variables

    def _set_variables(self, variables: Dict[str, Any]):
        super()._set_variables(variables)

        # VAE
        self._engine.set_variables(variables['engine'])

        # Learning Manager
        if 'learning_manager' in variables.keys():
            self.learning_manager = LearningManager()
            self.learning_manager.set_variables(variables['learning_manager'])

    def export_model(self, fp: BinaryIO, title: str = 'HighDimSynthesizer', description: str = None, author: str = None):
        """Save HighDimSynthesizer to file.

        Args:
            fp (BinaryIO): File object able to write bytes-like objects.
            title (str, optional): Identifier for this synthesizer. Defaults to 'HighDimSynthesizer'
            description (str, optional): Metadata. Defaults to None.
            author (str, optional): Author metadata. Defaults to None.

        Examples:

            Open binary file and save ``HighDimSynthesizer``:

            >>> with open('synthesizer.bin', 'wb') as f:
                    HighDimSynthesizer.export_model(f)
        """
        title = 'HighDimSynthesizer' if title is None else title
        author = 'SDK-v{}'.format(__version__) if author is None else author

        variables = self._get_variables()

        model_binary = ModelBinary(
            body=pickle.dumps(variables),
            title=title,
            description=description,
            author=author
        )
        model_binary.serialize(fp)

    @staticmethod
    def import_model(fp: BinaryIO):
        """Load HighDimSynthesizer from file.

        Args:
            fp (BinaryIO): File object able to read bytes-like objects.

        Examples:

            Open binary file and load ``HighDimSynthesizer``:

            >>> with open('synthesizer.bin', 'rb') as f:
                    synthesizer = HighDimSynthesizer.import_model(f)
        """

        model_binary = ModelBinary()
        model_binary.deserialize(fp)

        body = model_binary.get_body()
        if body is None:
            raise ValueError("The body of the given Binary Model is empty")
        variables = pickle.loads(model_binary.get_body())

        return HighDimSynthesizer._from_dict(variables)

    @staticmethod
    def _from_dict(variables: dict, summarizer_dir: str = None, summarizer_name: str = None):
        synth = HighDimSynthesizer.__new__(HighDimSynthesizer)
        super(HighDimSynthesizer, synth).__init__(
            name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name
        )

        synth.config = HighDimConfig(**variables['config'])

        # Dataframes
        synth.df_meta = DataFrameMeta.from_dict(variables['df_meta'])
        synth._df_model = DataFrameModel.from_dict(variables['df_model'])
        synth._df_model_independent = DataFrameModel.from_dict(variables['df_model_independent'])
        synth._df_value = DataFrameValue.from_dict(variables['df_value'])
        synth._df_transformer = DataFrameTransformer.from_dict(variables['df_transformer'])

        # VAE
        synth._engine = HighDimEngine(
            name='vae', df_value=synth._df_value, config=EngineConfig(
                latent_size=variables['latent_size'], network=variables['network'], capacity=variables['capacity'],
                num_layers=variables['num_layers'], residual_depths=variables['residual_depths'],
                batch_norm=variables['batch_norm'], activation=variables['activation'],
                optimizer=variables['optimizer'], learning_rate=variables['learning_rate'],
                decay_steps=variables['decay_steps'], decay_rate=variables['decay_rate'],
                initial_boost=variables['initial_boost'], clip_gradients=variables['clip_gradients'],
                beta=variables['beta'], weight_decay=variables['weight_decay']
            )
        )
        synth._engine.set_variables(variables['engine'])

        # Learning Manager
        synth._learning_manager = None
        if 'learning_manager' in variables.keys():
            synth._learning_manager = LearningManager()
            synth._learning_manager.set_variables(variables['learning_manager'])

        return synth

    # alias method for learn
    fit = learn

    # alias method for synthesize
    generate = synthesize
