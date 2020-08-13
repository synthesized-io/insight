from typing import Union, List, Optional, Callable

from dataclasses import dataclass, fields
import pandas as pd


@dataclass
class AddressParams:
    postcode_label: Union[str, List[str], None] = None
    county_label: Union[str, List[str], None] = None
    city_label: Union[str, List[str], None] = None
    district_label: Union[str, List[str], None] = None
    street_label: Union[str, List[str], None] = None
    house_number_label: Union[str, List[str], None] = None
    flat_label: Union[str, List[str], None] = None
    house_name_label: Union[str, List[str], None] = None


@dataclass
class AddressMetaConfig:
    addresses_file: Optional[str] = '~/.synthesized/addresses.jsonl.gz'

    @property
    def address_meta_config(self):
        return AddressMetaConfig(**{f.name: self.__getattribute__(f.name) for f in fields(AddressMetaConfig)})


@dataclass
class BankParams:
    bic_label: Union[str, List[str], None] = None
    sort_code_label: Union[str, List[str], None] = None
    account_label: Union[str, List[str], None] = None


@dataclass
class CompoundAddressParams:
    address_label: Optional[str] = None
    postcode_regex: Optional[str] = None


@dataclass
class PersonParams:
    title_label: Union[str, List[str], None] = None
    gender_label: Union[str, List[str], None] = None
    name_label: Union[str, List[str], None] = None
    firstname_label: Union[str, List[str], None] = None
    lastname_label: Union[str, List[str], None] = None
    email_label: Union[str, List[str], None] = None
    mobile_number_label: Union[str, List[str], None] = None
    home_number_label: Union[str, List[str], None] = None
    work_number_label: Union[str, List[str], None] = None


@dataclass
class PersonMetaConfig:
    dict_cache_size: int = 10000

    @property
    def person_meta_config(self):
        return PersonMetaConfig(**{f.name: self.__getattribute__(f.name) for f in fields(PersonMetaConfig)})


@dataclass
class MetaExtractorConfig(AddressMetaConfig, PersonMetaConfig):
    categorical_threshold_log_multiplier: float = 2.5
    parsing_nan_fraction_threshold: float = 0.25

    @property
    def value_factory_config(self):
        return MetaExtractorConfig(**{f.name: self.__getattribute__(f.name) for f in fields(MetaExtractorConfig)})


@dataclass
class CategoricalConfig:
    categorical_weight: float = 3.5
    temperature: float = 1.0
    moving_average: bool = True

    @property
    def categorical_config(self):
        return CategoricalConfig(**{f.name: self.__getattribute__(f.name) for f in fields(CategoricalConfig)})


@dataclass
class ContinuousConfig:
    continuous_weight: float = 5.0

    @property
    def continuous_config(self):
        return ContinuousConfig(**{f.name: self.__getattribute__(f.name) for f in fields(ContinuousConfig)})


@dataclass
class DecomposedContinuousConfig(ContinuousConfig):
    low_freq_weight: float = 1.0
    high_freq_weight: float = 1.0

    @property
    def decomposed_continuous_config(self):
        return DecomposedContinuousConfig(
            **{f.name: self.__getattribute__(f.name) for f in fields(DecomposedContinuousConfig)}
        )


@dataclass
class IdentifierConfig:
    capacity: int = 128

    @property
    def identifier_config(self):
        return IdentifierConfig(**{f.name: self.__getattribute__(f.name) for f in fields(IdentifierConfig)})


@dataclass
class NanConfig:
    nan_weight: float = 1.0

    @property
    def nan_config(self):
        return NanConfig(**{f.name: self.__getattribute__(f.name) for f in fields(NanConfig)})


@dataclass
class ValueFactoryConfig(CategoricalConfig, NanConfig, IdentifierConfig, DecomposedContinuousConfig):
    capacity: int = 128
    produce_nans: bool = False

    @property
    def value_factory_config(self):
        return ValueFactoryConfig(**{f.name: self.__getattribute__(f.name) for f in fields(ValueFactoryConfig)})


@dataclass
class LearningManagerConfig:
    """
    max_training_time: Maximum training time in seconds (LearningManager)
    custom_stop_metric: Custom stop metric for LearningManager.
    """
    check_frequency: int = 100
    use_checkpointing: bool = True
    checkpoint_path: Optional[str] = None
    n_checks_no_improvement: int = 10
    max_to_keep: int = 3
    patience: int = 750
    tol: float = 1e-4
    must_reach_metric: Optional[float] = None
    good_enough_metric: Optional[float] = None
    stop_metric_name: Union[str, List[str], None] = None
    sample_size: Optional[int] = 10_000
    use_engine_loss: bool = True
    max_training_time: Optional[float] = None
    custom_stop_metric: Optional[Callable[[pd.DataFrame, pd.DataFrame], float]] = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @property
    def learning_manager_config(self):
        return LearningManagerConfig(**{f.name: self.__getattribute__(f.name) for f in fields(LearningManagerConfig)})


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
    beta: beta.
    weight_decay: Weight decay.
    learning_manager: Whether to use LearningManager.
    """
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


@dataclass
class SeriesConfig(ValueFactoryConfig, LearningManagerConfig):
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
    beta: beta.
    weight_decay: Weight decay.
    learning_manager: Whether to use LearningManager.
    """
    distribution: str = 'normal'
    latent_size: int = 128
    # Network
    network: str = 'mlp'
    capacity: int = 128
    num_layers: int = 2
    residual_depths: Union[None, int, List[int]] = None
    batch_norm: bool = False
    activation: str = 'leaky_relu'
    # Optimizer
    optimizer: str = 'adam'
    learning_rate: float = 3e-3
    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    initial_boost: int = 0
    clip_gradients: float = 1.0
    # Batch size
    batch_size: int = 32
    increase_batch_size_every: Optional[int] = 500
    max_batch_size: Optional[int] = None
    # Losses
    beta: float = 0.01
    weight_decay: float = 1e-6
    learning_manager: bool = False
    # Series
    lstm_mode: str = 'rdssm'
    max_seq_len: int = 512
    series_dropout: float = 0.5
