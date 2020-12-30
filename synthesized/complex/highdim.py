"""This module implements the BasicSynthesizer class."""
import logging
from typing import Callable, List, Union, Dict, Optional, Tuple, Any, BinaryIO, Sequence

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from .binary_builder import ModelBinary

from ..common.generative import HighDimEngine
from ..common.learning_manager import LearningManager
from ..common.synthesizer import Synthesizer
from ..common.util import record_summaries_every_n_global_steps
from ..common.values import Value, ValueExtractor, DataFrameValue
from ..config import HighDimConfig
from ..metadata_new import DataFrameMeta
from ..transformer import DataFrameTransformer
from ..version import __version__

logger = logging.getLogger(__name__)


class HighDimSynthesizer(Synthesizer):
    """The main synthesizer implementation.

    Synthesizer which can learn from data to produce basic tabular data with independent rows, that
    is, no temporal or otherwise conditional relation between the rows.
    """

    def __init__(
            self, df_meta: DataFrameMeta, conditions: List[str] = None, summarizer_dir: str = None,
            summarizer_name: str = None, config: HighDimConfig = HighDimConfig(),
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
        self.df_value: DataFrameValue = ValueExtractor.extract(
            df_meta=df_meta, name='data_frame_value', conditions=conditions, config=config.value_factory_config
        )
        self.df_transformer: DataFrameTransformer = DataFrameTransformer.from_meta(self.df_meta)

        # VAE
        self.engine = HighDimEngine(
            name='vae', df_value=self.df_value, conditions=self.get_conditions(),
            latent_size=config.latent_size, network=config.network, capacity=config.capacity,
            num_layers=config.num_layers, residual_depths=config.residual_depths, batch_norm=config.batch_norm,
            activation=config.activation,
            optimizer=config.optimizer, learning_rate=config.learning_rate, decay_steps=config.decay_steps,
            decay_rate=config.decay_rate, initial_boost=config.initial_boost, clip_gradients=config.clip_gradients,
            beta=config.beta, weight_decay=config.weight_decay
        )

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

    # Not sure this is needed now we have self.df_value
    # def get_values(self) -> List[Value]:
    #     return self.df_value

    # TODO: implement conditions
    def get_conditions(self) -> List[Value]:
        return []

    def get_data_feed_dict(self, df: pd.DataFrame) -> Dict[str, Sequence[tf.Tensor]]:
        data: Dict[str, Sequence[tf.Tensor]] = {
            name: tuple([
                tf.constant(df[column])
                for column in value.columns()
            ])
            for name, value in self.df_value.items()
        }
        return data

    def get_conditions_feed_dict(self, df_conditions: pd.DataFrame) -> Dict[str, Sequence[tf.Tensor]]:
        data: Dict[str, Sequence[tf.Tensor]] = {
            value: tuple([
                tf.constant(df_conditions[df_name])
                for df_name in self.df_value[value].df_names
            ])
            for value in self.get_conditions()
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
            conditions=[value.specification() for value in self.value_factory.get_conditions()],
            engine=self.engine.specification(), batch_size=self.batch_size
        )
        return spec

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

        num_data = len(df_train)
        if not self.df_transformer._fitted:
            self.df_transformer.fit(df_train)
        df_train_pre = self.df_transformer.transform(df_train)
        if self.df_value.learned_input_size() == 0:
            return

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

                    losses = self.get_losses()
                    if callback(self, iteration, losses) is True:
                        return
                else:
                    self.engine.learn(xs=feed_dict)

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
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None, produce_nans: bool = False,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            produce_nans: Whether to produce NaNs.
            progress_callback: Progress bar callback.

        Returns:
            The generated data.

        """
        if progress_callback is not None:
            progress_callback(0)

        if num_rows <= 0:
            raise ValueError("Given 'num_rows' must be greater than zero, given '{}'.".format(num_rows))

        if type(conditions) is dict:
            conditions = pd.DataFrame(index=np.arange(num_rows), data=conditions)

        df_conditions = None
        columns = self.df_meta.columns

        if df_conditions is not None and len(df_conditions) > 0:
            conditions_dict = self.get_conditions_feed_dict(df_conditions=df_conditions)
            conditions_dataset = tf.data.Dataset.from_tensor_slices(conditions_dict).repeat(count=None)
        else:
            conditions_dataset = None

        if len(columns) == 0:
            return pd.DataFrame([[], ] * num_rows)

        if self.df_value.learned_input_size() == 0:
            return self.df_transformer.inverse_transform(pd.DataFrame([[], ] * num_rows))

        if self.writer is not None:
            tf.summary.trace_on(graph=True, profiler=False)

        if self.synthesis_batch_size is None or self.synthesis_batch_size > num_rows:
            synthesized = None
            if conditions_dataset is None:
                synthesized = self.engine.synthesize(tf.constant(num_rows, dtype=tf.int64), cs=dict(),
                                                     produce_nans=produce_nans)
            else:
                for cs in conditions_dataset.batch(batch_size=num_rows).take(1):
                    synthesized = self.engine.synthesize(tf.constant(num_rows, dtype=tf.int64), cs=cs,
                                                         produce_nans=produce_nans)
            assert synthesized is not None
            synthesized = self.df_value.split_outputs(synthesized)
            df_synthesized = pd.DataFrame.from_dict(synthesized)
            if progress_callback is not None:
                progress_callback(98)

        else:
            dict_synthesized = None
            if num_rows % self.synthesis_batch_size > 0:
                synthesized = None
                if conditions_dataset is None:
                    synthesized = self.engine.synthesize(
                        tf.constant(num_rows % self.synthesis_batch_size, dtype=tf.int64), cs=dict(),
                        produce_nans=produce_nans
                    )
                else:
                    for cs in conditions_dataset.batch(batch_size=num_rows % self.synthesis_batch_size).take(1):
                        synthesized = self.engine.synthesize(
                            tf.constant(num_rows % self.synthesis_batch_size, dtype=tf.int64), cs=cs,
                            produce_nans=produce_nans
                        )
                assert synthesized is not None
                dict_synthesized = self.df_value.split_outputs(synthesized)
                dict_synthesized = {k: v.tolist() for k, v in dict_synthesized.items()}

            n_batches = num_rows // self.synthesis_batch_size

            if conditions_dataset:
                conditions_dataset = conditions_dataset.batch(batch_size=self.synthesis_batch_size).take(n_batches)
                for k, cs in enumerate(conditions_dataset):
                    other = self.engine.synthesize(tf.constant(self.synthesis_batch_size, dtype=tf.int64), cs=cs,
                                                   produce_nans=produce_nans)
                    other = self.df_value.split_outputs(other)
                    if dict_synthesized is None:
                        dict_synthesized = other
                        dict_synthesized = {key: v.tolist() for key, v in dict_synthesized.items()}
                    else:
                        for c in other.keys():
                            dict_synthesized[c].extend(other[c].tolist())

                    if progress_callback is not None:
                        # report approximate progress from 0% to 98% (2% are reserved for post actions)
                        progress_callback(round((k + 1) * 98.0 / n_batches))
            else:
                for k in range(n_batches):
                    other = self.engine.synthesize(tf.constant(self.synthesis_batch_size, dtype=tf.int64), cs=dict(),
                                                   produce_nans=produce_nans)
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

        df_synthesized = self.df_transformer.inverse_transform(df_synthesized)[columns]

        if self.writer is not None:
            tf.summary.trace_export(name='Synthesize', step=0)
            tf.summary.trace_off()

        if progress_callback is not None:
            progress_callback(100)

        return df_synthesized

    def encode(
            self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None, produce_nans: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encodes dataset and returns the corresponding latent space and generated data.

        Args:
            df_encode: Input dataset.
            conditions: The condition values for the generated rows.
            produce_nans: Whether to produce NaNs.

        Returns:
            (Pandas DataFrame of latent space, Pandas DataFrame of decoded space) corresponding to input data
        """
        df_encode = df_encode.copy()
        df_encode = self.df_transformer.transform(df=df_encode)

        num_rows = len(df_encode)
        data = self.get_data_feed_dict(df_encode)
        if conditions is not None and len(conditions) > 0:
            conditions_dict = self.get_conditions_feed_dict(df_conditions=conditions)
            conditions_dataset = tf.data.Dataset.from_tensor_slices(conditions_dict).repeat(count=None)
        else:
            conditions_dataset = None

        if conditions_dataset is not None:
            encoded, decoded = None, None
            for cs in conditions_dataset.batch(num_rows).take(1):
                encoded, decoded = self.engine.encode(xs=data, cs=cs, produce_nans=produce_nans)
            assert encoded is not None and decoded is not None
        else:
            encoded, decoded = self.engine.encode(xs=data, cs=dict(), produce_nans=produce_nans)

        columns = np.concatenate([c.learned_output_columns() for c in self.df_value.values()])
        decoded = self.df_value.split_outputs(decoded)
        df_synthesized = pd.DataFrame.from_dict(decoded)[columns]
        df_synthesized = self.postprocess(df=df_synthesized)

        latent = np.concatenate((encoded['sample'], encoded['mean'], encoded['std']), axis=1)
        df_encoded = pd.DataFrame.from_records(latent, columns=[f"{ls}_{n}" for ls in 'lms'
                                                                for n in range(encoded['sample'].shape[1])])

        return df_encoded, df_synthesized

    def encode_deterministic(
            self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None, produce_nans: bool = False
    ) -> pd.DataFrame:
        """Deterministically encodes a dataset and returns it with imputed nans.

        Args:
            df_encode: Input dataset
            conditions: The condition values for the generated rows.
            produce_nans: Whether to produce NaNs.

        Returns:
            Pandas DataFrame of decoded space corresponding to input data
        """
        df_encode = self.df_transformer.transform(df=df_encode)

        num_rows = len(df_encode)
        data = self.get_data_feed_dict(df_encode)
        if conditions is not None and len(conditions) > 0:
            conditions_dict = self.get_conditions_feed_dict(df_conditions=conditions)
            conditions_dataset = tf.data.Dataset.from_tensor_slices(conditions_dict).repeat(count=None)
        else:
            conditions_dataset = None

        if conditions_dataset is not None:
            decoded = None
            for cs in conditions_dataset.batch(num_rows).take(1):
                decoded = self.engine.encode_deterministic(xs=data, cs=cs, produce_nans=produce_nans)
            assert decoded is not None
        else:
            decoded = self.engine.encode_deterministic(xs=data, cs=dict(), produce_nans=produce_nans)

        decoded = self.df_value.split_outputs(decoded)
        columns = self.df_meta.columns
        df_synthesized = pd.DataFrame.from_dict(decoded)[columns]
        df_synthesized = self.postprocess(df=df_synthesized)

        return df_synthesized.set_index(df_encode.index)

    def get_variables(self) -> Dict[str, Any]:
        variables = super().get_variables()
        variables.update(
            # Data Panel TODO: get variables for values and metas
            # df_meta=self.df_meta.get_variables(),
            # Value Factory
            df_value=self.df_value.get_variables(),

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

        raise NotImplementedError("from dict functionality still needs to be implemented")
        # Data Panel
        synth.df_meta = DataFrameMeta.from_dict(variables['df_meta'])

        # Value Factory TODO: replace with df_value
        # synth.value_factory = ValueFactory.from_dict(variables['value_factory'])

        # VAE
        synth.engine = HighDimEngine(
            name='vae', values=synth.get_values(), conditions=synth.get_conditions(),
            latent_size=variables['latent_size'], network=variables['network'], capacity=variables['capacity'],
            num_layers=variables['num_layers'], residual_depths=variables['residual_depths'],
            batch_norm=variables['batch_norm'], activation=variables['activation'], optimizer=variables['optimizer'],
            learning_rate=variables['learning_rate'], decay_steps=variables['decay_steps'],
            decay_rate=variables['decay_rate'], initial_boost=variables['initial_boost'],
            clip_gradients=variables['clip_gradients'], beta=variables['beta'], weight_decay=variables['weight_decay']
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
