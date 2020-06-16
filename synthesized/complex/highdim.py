"""This module implements the BasicSynthesizer class."""
import logging
from typing import Callable, List, Union, Dict, Optional, Tuple, Any, BinaryIO

from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from .binary_builder import ModelBinary
from ..metadata import DataPanel, ValueMeta
from ..common.generative import VAEOld
from ..common.learning_manager import LearningManager, LearningManagerConfig
from ..common.synthesizer import Synthesizer
from ..common.util import record_summaries_every_n_global_steps
from ..common.values import Value, ValueFactory, ValueFactoryConfig, ValueFactoryWrapper

from ..version import __version__

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)


@dataclass
class HighDimConfig(ValueFactoryConfig, LearningManagerConfig):
    """
    distribution: Distribution type: "normal".
    latent_size: Latent size.
    network: Network type: "mlp" or "resnet".
    capacity: Architecture capacity.
    num_layers: Architecture depth.
    residual_depths: The depth(s) of each individual residual layer.
    batch_norm: Whether to use batch normalization.
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
    learning_manager: Whether to use LearningManager.
    """
    # VAE distribution
    distribution: str = 'normal'
    latent_size: int = 32
    # Network
    network: str = 'resnet'
    capacity: int = 128
    num_layers: int = 2
    residual_depths: Union[None, int, List[int]] = 6
    batch_norm: bool = True
    activation: str = 'relu'
    # Optimizer
    optimizer: str = 'adam'
    learning_rate: float = 3e-3
    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    initial_boost: int = 0
    clip_gradients: float = 1.0
    # Batch size
    batch_size: int = 64
    increase_batch_size_every: Optional[int] = 500
    max_batch_size: Optional[int] = 1024
    synthesis_batch_size: Optional[int] = 16384
    # Losses
    beta: float = 1.0
    weight_decay: float = 1e-3
    learning_manager: bool = True


class HighDimSynthesizer(Synthesizer):
    """The main synthesizer implementation.

    Synthesizer which can learn from data to produce basic tabular data with independent rows, that
    is, no temporal or otherwise conditional relation between the rows.
    """

    def __init__(
        self, data_panel: DataPanel, conditions: List[str] = None, summarizer_dir: str = None,
        summarizer_name: str = None, config: HighDimConfig = HighDimConfig(),
    ):
        """Initialize a new BasicSynthesizer instance.

        Args:
            data_panel: Data sample which summarizes all relevant characteristics,
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

        self.data_panel = data_panel

        self.value_factory = ValueFactory(
            data_panel=data_panel, name='value_factory',  conditions=conditions, config=config.value_factory_config
        )

        # VAE
        self.vae = VAEOld(
            name='vae', values=self.get_values(), conditions=self.get_conditions(),
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
            use_vae_loss = False if config.custom_stop_metric else True
            self.learning_manager = LearningManager(
                max_training_time=config.max_training_time, use_vae_loss=use_vae_loss,
                custom_stop_metric=config.custom_stop_metric
            )

    def get_values(self) -> List[Value]:
        return self.value_factory.get_values()

    def get_conditions(self) -> List[Value]:
        return self.value_factory.get_conditions()

    def get_value_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return [(v, self.data_panel[v.name]) for v in self.value_factory.all_values]

    def get_condition_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return [(v, self.data_panel[v.name]) for v in self.value_factory.get_conditions()]

    def get_losses(self, data: Dict[str, tf.Tensor]) -> tf.Tensor:
        self.vae.xs = data
        self.vae.loss()
        return self.vae.losses

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            values=[value.specification() for value in self.value_factory.get_values()],
            conditions=[value.specification() for value in self.value_factory.get_conditions()],
            vae=self.vae.specification(), batch_size=self.batch_size
        )
        return spec

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.data_panel.preprocess(df)
        return df

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[Synthesizer, int, dict], bool] = Synthesizer.logging,
        callback_freq: int = 0, low_memory: bool = False
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
            low_memory: If set to true, each batch will be preprocessed in each iteration, otherwise a copy of
                df_train will be preprocessed at once at the start.

        """
        assert num_iterations or self.learning_manager, "'num_iterations' must be set if learning_manager=False"

        if self.learning_manager:
            self.learning_manager.restart_learning_manager()
            self.learning_manager.set_check_frequency(self.batch_size)

        num_data = len(df_train)

        if not low_memory:
            df_train = df_train.copy()
            df_train_pre = self.data_panel.preprocess(df_train.copy())
        else:
            sample_size_lm = min(self.learning_manager.sample_size,
                                 num_data) if self.learning_manager and self.learning_manager.sample_size else num_data

        with record_summaries_every_n_global_steps(callback_freq, self.global_step):
            keep_learning = True
            iteration = 1
            while keep_learning:
                batch = tf.random.uniform(shape=(self.batch_size,), maxval=num_data, dtype=tf.int64)

                if low_memory:
                    feed_dict = self.get_data_feed_dict(self.data_panel.preprocess(df_train.iloc[batch].copy()))
                else:
                    feed_dict = self.get_data_feed_dict(df_train_pre.iloc[batch].copy())

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

                if self.learning_manager:
                    if low_memory:
                        batch = np.random.choice(num_data, size=sample_size_lm, replace=False)
                        data = self.get_data_feed_dict(self.data_panel.preprocess(df_train.iloc[batch].copy()))
                        if self.learning_manager.stop_learning(iteration, synthesizer=self,
                                                               data_dict=data, num_data=num_data,
                                                               df_train_orig=df_train.sample(sample_size_lm)):
                            break
                    else:
                        data = self.get_data_feed_dict(df_train_pre)
                        if self.learning_manager.stop_learning(iteration, synthesizer=self,
                                                               data_dict=data, num_data=num_data,
                                                               df_train_orig=df_train):
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
        if progress_callback is not None:
            progress_callback(0)

        if num_rows <= 0:
            raise ValueError("Given 'num_rows' must be greater than zero, given '{}'.".format(num_rows))

        df_conditions = self.data_panel.preprocess_by_name(conditions, [c.name for c in self.get_conditions()])
        columns = self.data_panel.columns

        if len(columns) == 0:
            return pd.DataFrame([[], ]*num_rows)

        if self.writer is not None:
            tf.summary.trace_on(graph=True, profiler=False)

        if self.synthesis_batch_size is None or self.synthesis_batch_size > num_rows:
            feed_dict = self.get_conditions_feed_dict(df_conditions, num_rows)
            synthesized = self.vae.synthesize(num_rows, cs=feed_dict)
            synthesized = self.data_panel.split_outputs(synthesized)

            df_synthesized = pd.DataFrame.from_dict(synthesized)[columns]
            if progress_callback is not None:
                progress_callback(98)

        else:
            feed_dict = self.get_conditions_feed_dict(df_conditions, num_rows)
            synthesized = self.vae.synthesize(tf.constant(num_rows % self.synthesis_batch_size, dtype=tf.int64),
                                              cs=feed_dict)
            synthesized = self.data_panel.split_outputs(synthesized)
            df_synthesized = pd.DataFrame.from_dict(synthesized)[columns]

            feed_dict = self.get_conditions_feed_dict(df_conditions, self.synthesis_batch_size)
            n_batches = num_rows // self.synthesis_batch_size
            conditions_data = self.get_conditions_data(df_conditions)

            for k in range(n_batches):
                if len(conditions_data) > 0:
                    feed_dict.update({
                        name: tf.constant(
                            condition_data[k * self.synthesis_batch_size: (k + 1) * self.synthesis_batch_size],
                            dtype=tf.float32)
                        for name, condition_data in conditions_data.items()
                        if condition_data.shape == (num_rows,)
                    })
                other = self.vae.synthesize(tf.constant(self.synthesis_batch_size, dtype=tf.int64), cs=feed_dict)
                other = self.data_panel.split_outputs(other)
                df_synthesized = df_synthesized.append(
                    pd.DataFrame.from_dict(other)[columns], ignore_index=True
                )
                if progress_callback is not None:
                    # report approximate progress from 0% to 98% (2% are reserved for post actions)
                    progress_callback(round((k + 1) * 98.0 / n_batches))

        df_synthesized = self.data_panel.postprocess(df_synthesized)

        if self.writer is not None:
            tf.summary.trace_export(name='Synthesize', step=0)
            tf.summary.trace_off()

        if progress_callback is not None:
            progress_callback(100)

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
        df_encode = self.data_panel.preprocess(df=df_encode)
        df_conditions = self.data_panel.preprocess_by_name(conditions, [c.name for c in self.get_conditions()])

        num_rows = len(df_encode)
        data = self.get_data_feed_dict(df_encode)
        conditions_data = self.get_conditions_feed_dict(df_conditions, num_rows=num_rows, batch_size=None)

        encoded, decoded = self.vae.encode(xs=data, cs=conditions_data)

        columns = self.data_panel.columns
        decoded = self.data_panel.split_outputs(decoded)
        df_synthesized = pd.DataFrame.from_dict(decoded)[columns]
        df_synthesized = self.data_panel.postprocess(df=df_synthesized)

        latent = np.concatenate((encoded['sample'], encoded['mean'], encoded['std']), axis=1)
        df_encoded = pd.DataFrame.from_records(latent, columns=[f"{ls}_{n}" for ls in 'lms'
                                                                for n in range(encoded['sample'].shape[1])])

        return df_encoded, df_synthesized

    def encode_deterministic(self, df_encode: pd.DataFrame,
                             conditions: Union[dict, pd.DataFrame] = None) -> pd.DataFrame:
        """Deterministically encodes a dataset and returns it with imputed nans.

        Args:
            df_encode: Input dataset
            conditions: The condition values for the generated rows.

        Returns:
            Pandas DataFrame of decoded space corresponding to input data
        """
        df_encode = df_encode.copy()
        df_encode = self.data_panel.preprocess(df=df_encode)
        df_conditions = self.data_panel.preprocess_by_name(conditions, [c.name for c in self.get_conditions()])

        num_rows = len(df_encode)
        data = self.get_data_feed_dict(df_encode)
        conditions_data = self.get_conditions_feed_dict(df_conditions, num_rows=num_rows, batch_size=None)

        decoded = self.vae.encode_deterministic(xs=data, cs=conditions_data)

        columns = self.data_panel.columns
        df_synthesized = pd.DataFrame.from_dict(decoded)[columns]
        df_synthesized = self.data_panel.postprocess(df=df_synthesized)

        return df_synthesized

    def get_variables(self) -> Dict[str, Any]:
        variables = super().get_variables()
        variables.update(
            # Data Panel
            data_panel=self.data_panel.get_variables(),
            # Value Factory
            value_factory=self.value_factory.get_variables(),

            # VAE
            vae=self.vae.get_variables(),
            latent_size=self.vae.latent_size,
            network=self.vae.network,
            capacity=self.vae.capacity,
            num_layers=self.vae.num_layers,
            residual_depths=self.vae.residual_depths,
            batch_norm=self.vae.batch_norm,
            activation=self.vae.activation,
            optimizer=self.vae.optimizer_name,
            learning_rate=self.vae.learning_rate,
            decay_steps=self.vae.decay_steps,
            decay_rate=self.vae.decay_rate,
            initial_boost=self.vae.initial_boost,
            clip_gradients=self.vae.clip_gradients,
            beta=self.vae.beta,
            weight_decay=self.vae.weight_decay,

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
        self.vae.set_variables(variables['vae'])

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

        return HighDimSynthesizerWrapper(variables)


class HighDimSynthesizerWrapper(HighDimSynthesizer):
    def __init__(self, variables, summarizer_dir: str = None, summarizer_name: str = None):
        super(HighDimSynthesizer, self).__init__(
            name='synthesizer', summarizer_dir=summarizer_dir, summarizer_name=summarizer_name
        )

        # Data Panel
        self.data_panel = DataPanel.from_dict(variables['data_panel'])

        # Value Factory
        self.value_factory = ValueFactoryWrapper(name='value_factory', variables=variables['value_factory'])

        # VAE
        self.vae = VAEOld(
            name='vae', values=self.get_values(), conditions=self.get_conditions(),
            latent_size=variables['latent_size'], network=variables['network'], capacity=variables['capacity'],
            num_layers=variables['num_layers'], residual_depths=variables['residual_depths'],
            batch_norm=variables['batch_norm'], activation=variables['activation'], optimizer=variables['optimizer'],
            learning_rate=variables['learning_rate'], decay_steps=variables['decay_steps'],
            decay_rate=variables['decay_rate'], initial_boost=variables['initial_boost'],
            clip_gradients=variables['clip_gradients'], beta=variables['beta'], weight_decay=variables['weight_decay']
        )
        self.vae.set_variables(variables['vae'])

        # Batch Sizes
        self.batch_size = variables['batch_size']
        self.increase_batch_size_every = variables['increase_batch_size_every']
        self.max_batch_size = variables['max_batch_size']
        self.synthesis_batch_size = variables['synthesis_batch_size']

        # Input argument placeholder for num_rows
        self.num_rows: Optional[tf.Tensor] = None

        # Learning Manager
        self.learning_manager: Optional[LearningManager] = None
        if 'learning_manager' in variables.keys():
            self.learning_manager = LearningManager()
            self.learning_manager.set_variables(variables['learning_manager'])
