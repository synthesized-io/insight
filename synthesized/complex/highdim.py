"""This module implements the BasicSynthesizer class."""
import logging
import pickle
from math import sqrt
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .binary_builder import ModelBinary
from ..common.generative import HighDimEngine
from ..common.learning_manager import LearningManager
from ..common.synthesizer import Synthesizer
from ..common.util import record_summaries_every_n_global_steps
from ..common.values import DataFrameValue, ValueExtractor
from ..config import EngineConfig, HighDimConfig
from ..metadata import DataFrameMeta
from ..model import ContinuousModel, DataFrameModel, DiscreteModel, Model
from ..model.factory import ModelFactory
from ..model.models import AddressModel, AssociatedHistogram, BankModel, FormattedStringModel, GenderModel, PersonModel
from ..transformer import DataFrameTransformer
from ..version import __version__

logger = logging.getLogger(__name__)


class HighDimSynthesizer(Synthesizer):
    """The main synthesizer implementation.

    Synthesizer which can learn from data to produce basic tabular data with independent rows, that
    is, no temporal or otherwise conditional relation between the rows.
    """

    def __init__(
            self, df_meta: DataFrameMeta, summarizer_dir: str = None,
            summarizer_name: str = None, config: HighDimConfig = HighDimConfig(),
            type_overrides: List[Union[ContinuousModel, DiscreteModel]] = None
    ):
        """Initialize a new BasicSynthesizer instance.

        Args:
            df_meta: Data sample which summarizes all relevant characteristics,
                so for instance all values a discrete-value column can take.
            summarizer_dir: Directory for TensorBoard summaries, automatically creates unique subfolder.

        """
        super(HighDimSynthesizer, self).__init__(
            name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name
        )
        self.batch_size = config.batch_size
        self.increase_batch_size_every = config.increase_batch_size_every
        self.max_batch_size: int = config.max_batch_size if config.max_batch_size else config.batch_size
        self.synthesis_batch_size = config.synthesis_batch_size

        self.df_meta: DataFrameMeta = df_meta
        self.df_model = ModelFactory()(df_meta, type_overrides=type_overrides)
        self.df_model_independent = self.split_df_model(self.df_model)

        self.df_value: DataFrameValue = ValueExtractor.extract(
            df_meta=self.df_model, name='data_frame_value', config=config.value_factory_config
        )
        self.df_transformer: DataFrameTransformer = DataFrameTransformer.from_meta(self.df_model)

        # VAE
        self.engine = HighDimEngine(name='vae', df_value=self.df_value, config=config.engine_config)

        # Input argument placeholder for num_rows
        self.num_rows: Optional[tf.Tensor] = None

        # Learning Manager
        self.learning_manager: Optional[LearningManager] = None
        if config.learning_manager:
            use_engine_loss = False if config.custom_stop_metric else True
            self.learning_manager = LearningManager(
                max_training_time=config.max_training_time, use_engine_loss=use_engine_loss,
                custom_stop_metric=config.custom_stop_metric, sample_size=1024
            )

    def get_data_feed_dict(self, df: pd.DataFrame) -> Dict[str, Sequence[tf.Tensor]]:
        data: Dict[str, Sequence[tf.Tensor]] = {
            name: tuple([
                tf.constant(df[column])
                for column in value.columns()
            ])
            for name, value in self.df_value.items()
        }
        return data

    def get_losses(self, data: Dict[str, Sequence[tf.Tensor]] = None) -> Dict[str, tf.Tensor]:
        if data is not None:
            self.engine.loss(data)

        losses = {
            'total-loss': self.engine.total_loss,
            'kl-loss': self.engine.kl_loss,
            'regularization-loss': self.engine.regularization_loss,
            'reconstruction-loss': self.engine.reconstruction_loss
        }
        return losses

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            values=[value.specification() for value in self.value_factory.get_values()],
            engine=self.engine.specification(), batch_size=self.batch_size
        )
        return spec

    def split_df_model(self, df_model: DataFrameModel) -> DataFrameModel:
        """Given a df_model, pop out those models that are not learned in the engine and
        return a df_model_independent containing these models.
        """
        df_meta_independent = DataFrameMeta(name='independent_models')
        df_model_independent = DataFrameModel(df_meta_independent)
        models_to_pop: List[str] = []
        models_to_add: List[Model] = []
        for name, model in df_model.items():
            if isinstance(model, ContinuousModel):
                continue

            if isinstance(model, AssociatedHistogram):
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
            self, df_train: pd.DataFrame, num_iterations: Optional[int],
            callback: Callable[[Synthesizer, int, dict], bool] = None,
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

        if self.learning_manager:
            self.learning_manager.restart_learning_manager()
            self.learning_manager.set_check_frequency(self.batch_size)

        if not self.df_meta._extracted:
            self.df_meta.extract(df_train)

        self.df_model_independent.fit(df=df_train)

        if not self.df_transformer.is_fitted():
            self.df_transformer.fit(df_train)

        df_train_pre = self.df_transformer.transform(df_train)
        if self.df_value.learned_input_size() == 0 and len(self.df_model) == 0:
            return

        num_data = len(df_train)
        with self.writer.as_default():
            with record_summaries_every_n_global_steps(callback_freq, self.global_step):
                keep_learning = True
                iteration = 0
                while keep_learning:
                    # Increment iteration number, and check if we reached max num_iterations
                    self.global_step.assign_add(1)
                    iteration += 1
                    if num_iterations:
                        keep_learning = iteration <= num_iterations
                    if keep_learning is False:
                        break

                    batch = tf.random.uniform(shape=(self.batch_size,), maxval=num_data, dtype=tf.int64)
                    feed_dict = self.get_data_feed_dict(df_train_pre.iloc[batch])

                    self.engine.learn(xs=feed_dict)

                    if callback is not None and callback_freq > 0 and (
                        iteration == 1 or iteration == num_iterations or iteration % callback_freq == 0
                    ):
                        losses = self.get_losses()
                        if callback(self, iteration, losses) is True:
                            return

                    if self.learning_manager:
                        if self.learning_manager.stop_learning(
                                iteration, synthesizer=self,
                        ):
                            break

                    # Increase batch size
                    tf.summary.scalar(name='batch_size', data=self.batch_size)
                    if self.increase_batch_size_every and iteration > 0 and self.batch_size < self.max_batch_size and \
                            iteration % self.increase_batch_size_every == 0:
                        self.batch_size *= 2
                        if self.batch_size > self.max_batch_size:
                            self.batch_size = self.max_batch_size

                        if self.batch_size == self.max_batch_size:
                            logger.info('Maximum batch size of {} reached.'.format(self.max_batch_size))
                        if self.learning_manager:
                            self.learning_manager.set_check_frequency(self.batch_size)

    def synthesize(
            self, num_rows: int, produce_nans: bool = False,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            produce_nans: Whether to produce NaNs.
            progress_callback: Progress bar callback.

        Returns:
            The generated data.

        """
        if progress_callback is not None:
            progress_callback(0)

        if num_rows <= 0:
            raise ValueError("Given 'num_rows' must be greater than zero, given '{}'.".format(num_rows))

        columns = self.df_meta.columns

        assert columns is not None
        if len(columns) == 0:
            return pd.DataFrame([[], ] * num_rows)

        if self.df_value.learned_input_size() > 0:
            if self.synthesis_batch_size is None or self.synthesis_batch_size > num_rows:
                synthesized = self.engine.synthesize(tf.constant(num_rows, dtype=tf.int64))
                assert synthesized is not None
                synthesized = self.df_value.split_outputs(synthesized)
                df_synthesized = pd.DataFrame.from_dict(synthesized)
                if progress_callback is not None:
                    progress_callback(98)

            else:
                dict_synthesized = None
                if num_rows % self.synthesis_batch_size > 0:
                    synthesized = self.engine.synthesize(
                        tf.constant(num_rows % self.synthesis_batch_size, dtype=tf.int64)
                    )
                    assert synthesized is not None
                    dict_synthesized = self.df_value.split_outputs(synthesized)
                    dict_synthesized = {k: v.tolist() for k, v in dict_synthesized.items()}

                n_batches = num_rows // self.synthesis_batch_size

                for k in range(n_batches):
                    other = self.engine.synthesize(tf.constant(self.synthesis_batch_size, dtype=tf.int64))
                    other = self.df_value.split_outputs(other)
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

            df_synthesized = self.df_transformer.inverse_transform(df_synthesized, produce_nans=produce_nans)

        else:
            df_synthesized = pd.DataFrame([[], ] * num_rows)

        df_sampled = self.df_model_independent.sample(n=num_rows, produce_nans=produce_nans, conditions=df_synthesized)
        df_independent = df_sampled[[c for c in df_sampled.columns if c not in df_synthesized.columns]]

        df_synthesized = pd.concat((df_synthesized, df_independent), axis=1)
        df_synthesized = df_synthesized[columns]

        if progress_callback is not None:
            progress_callback(100)

        return df_synthesized

    def encode(
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
        df_encode = self.df_transformer.transform(df=df_encode)

        data = self.get_data_feed_dict(df_encode)
        encoded, decoded = self.engine.encode(xs=data)

        decoded = self.df_value.split_outputs(decoded)
        df_synthesized = pd.DataFrame.from_dict(decoded)
        df_synthesized = self.df_transformer.inverse_transform(df_synthesized, produce_nans=produce_nans)

        df_sampled = self.df_model_independent.sample(
            n=len(df_synthesized), produce_nans=produce_nans, conditions=df_synthesized)
        df_independent = df_sampled[[c for c in df_sampled.columns if c not in df_synthesized.columns]]
        df_synthesized = pd.concat((df_synthesized, df_independent), axis=1)
        df_synthesized = df_synthesized[self.df_meta.columns]

        latent = np.concatenate((encoded['sample'], encoded['mean'], encoded['std']), axis=1)
        df_encoded = pd.DataFrame.from_records(latent, columns=[f"{ls}_{n}" for ls in 'lms'
                                                                for n in range(encoded['sample'].shape[1])])

        return df_encoded, df_synthesized

    def encode_deterministic(
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
        df_encode = self.df_transformer.transform(df=df_encode.copy().reset_index(drop=True))
        data = self.get_data_feed_dict(df_encode)
        decoded = self.engine.encode_deterministic(xs=data)
        decoded = self.df_value.split_outputs(decoded)
        df_synthesized = pd.DataFrame.from_dict(decoded)
        df_synthesized = self.df_transformer.inverse_transform(df_synthesized, produce_nans=produce_nans)

        df_independent: pd.DataFrame = df_orig.reset_index(drop=True)[
            [c for c in df_orig.columns if c not in df_synthesized.columns]]
        if not produce_nans:
            df_sampled = self.df_model_independent.sample(
                n=len(df_synthesized), produce_nans=False, conditions=df_synthesized)
            df_sampled = df_sampled[[c for c in df_independent.columns]]
            df_independent = df_independent.where(df_independent.notna(), other=df_sampled)

        df_synthesized = pd.concat((df_synthesized, df_independent), axis=1)

        df_synthesized = df_synthesized[self.df_meta.columns]
        df_synthesized.index = df_orig.index.copy()
        return df_synthesized

    def get_variables(self) -> Dict[str, Any]:
        variables = super().get_variables()
        variables.update(
            df_meta=self.df_meta.to_dict(),
            df_model=self.df_model.to_dict(),
            df_model_independent=self.df_model_independent.to_dict(),
            df_value=self.df_value.to_dict(),
            df_transformer=self.df_transformer.to_dict(),

            # VAE
            engine=self.engine.get_variables(),
            latent_size=self.engine.latent_size,
            network=self.engine.network,
            capacity=self.engine.capacity,
            num_layers=self.engine.num_layers,
            residual_depths=self.engine.residual_depths,
            batch_norm=self.engine.batch_norm,
            activation=self.engine.activation,
            optimizer=self.engine.optimizer_name,
            learning_rate=self.engine.learning_rate,
            decay_steps=self.engine.decay_steps,
            decay_rate=self.engine.decay_rate,
            initial_boost=self.engine.initial_boost,
            clip_gradients=self.engine.clip_gradients,
            beta=self.engine.beta,
            weight_decay=self.engine.weight_decay,

            # HighDim
            batch_size=self.batch_size,
            increase_batch_size_every=self.increase_batch_size_every,
            max_batch_size=self.max_batch_size,
            synthesis_batch_size=self.synthesis_batch_size,

            # Learning Manager
            learning_manager=self.learning_manager.get_variables() if self.learning_manager else None
        )

        return variables

    def set_variables(self, variables: Dict[str, Any]):
        super().set_variables(variables)

        # Value Factory
        self.value_factory.set_variables(variables['value_factory'])

        # VAE
        self.engine.set_variables(variables['engine'])

        # Batch Sizes
        self.batch_size = variables['batch_size']
        self.increase_batch_size_every = variables['increase_batch_size_every']
        self.max_batch_size = variables['max_batch_size']
        self.synthesis_batch_size = variables['synthesis_batch_size']

        # Learning Manager
        if 'learning_manager' in variables.keys():
            self.learning_manager = LearningManager()
            self.learning_manager.set_variables(variables['learning_manager'])

    def export_model(self, fp: BinaryIO, title: str = None, description: str = None, author: str = None):
        title = 'HighDimSynthesizer' if title is None else title
        description = None if title is None else description
        author = 'SDK-v{}'.format(__version__) if title is None else author

        variables = self.get_variables()

        model_binary = ModelBinary(
            body=pickle.dumps(variables),
            title=title,
            description=description,
            author=author
        )
        model_binary.serialize(fp)

    @staticmethod
    def import_model(fp: BinaryIO):

        model_binary = ModelBinary()
        model_binary.deserialize(fp)

        body = model_binary.get_body()
        if body is None:
            raise ValueError("The body of the given Binary Model is empty")
        variables = pickle.loads(model_binary.get_body())

        return HighDimSynthesizer.from_dict(variables)

    @staticmethod
    def from_dict(variables: dict, summarizer_dir: str = None, summarizer_name: str = None):
        synth = HighDimSynthesizer.__new__(HighDimSynthesizer)
        super(HighDimSynthesizer, synth).__init__(
            name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name
        )

        # Dataframes
        synth.df_meta = DataFrameMeta.from_dict(variables['df_meta'])
        synth.df_model = DataFrameModel.from_dict(variables['df_model'])
        synth.df_model_independent = DataFrameModel.from_dict(variables['df_model_independent'])
        synth.df_value = DataFrameValue.from_dict(variables['df_value'])
        synth.df_transformer = DataFrameTransformer.from_dict(variables['df_transformer'])

        # VAE
        synth.engine = HighDimEngine(
            name='vae', df_value=synth.df_value, config=EngineConfig(
                latent_size=variables['latent_size'], network=variables['network'], capacity=variables['capacity'],
                num_layers=variables['num_layers'], residual_depths=variables['residual_depths'],
                batch_norm=variables['batch_norm'], activation=variables['activation'],
                optimizer=variables['optimizer'], learning_rate=variables['learning_rate'],
                decay_steps=variables['decay_steps'], decay_rate=variables['decay_rate'],
                initial_boost=variables['initial_boost'], clip_gradients=variables['clip_gradients'],
                beta=variables['beta'], weight_decay=variables['weight_decay']
            )
        )
        synth.engine.set_variables(variables['engine'])

        # Batch Sizes
        synth.batch_size = variables['batch_size']
        synth.increase_batch_size_every = variables['increase_batch_size_every']
        synth.max_batch_size = variables['max_batch_size']
        synth.synthesis_batch_size = variables['synthesis_batch_size']

        # Input argument placeholder for num_rows
        synth.num_rows = None

        # Learning Manager
        synth.learning_manager = None
        if 'learning_manager' in variables.keys():
            synth.learning_manager = LearningManager()
            synth.learning_manager.set_variables(variables['learning_manager'])

        return synth
