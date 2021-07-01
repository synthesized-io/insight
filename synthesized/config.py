import importlib
from dataclasses import dataclass, field, fields
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

# Meta Config Classes ----------------------------------------


@dataclass(repr=True)
class AddressRecord:
    full_address: Optional[str] = field(default=None, init=True, repr=False)
    postcode: Optional[str] = None
    county: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    street: Optional[str] = None
    house_number: Optional[str] = None
    flat: Optional[str] = None
    house_name: Optional[str] = None

    def __post_init__(self):
        for f in fields(self):
            if f.name != 'full_address':
                field_val = getattr(self, f.name)
                setattr(self, f.name, field_val.replace("'", "") if field_val is not None else None)
        if self.full_address is None:
            self.full_address = self.compute_full_address()

    def compute_full_address(self) -> str:
        address_str = ""

        if self.flat:
            address_str += f"{self.flat} "

        if self.house_number:
            address_str += f"{self.house_number} "

        if self.house_name:
            address_str += f"{self.house_name}, "

        if self.street:
            address_str += f"{self.street}, "

        if self.district:
            address_str += f"{self.district}, "

        if self.postcode:
            address_str += f"{self.postcode} "

        if self.city:
            address_str += f"{self.city} "

        if self.county:
            address_str += f"{self.county}"

        return address_str


@dataclass
class AnnotationParams:
    name: str


@dataclass(frozen=True)
class AddressLabels:
    """Column labels for the Address annotation.

    See also:

        :class:`~synthesized.metadata.value.Address` : Annotate a dataset with address information.
    """
    postcode_label: Optional[str] = None
    """Name of column with postcodes."""
    county_label: Optional[str] = None
    """Name of column with counties."""
    city_label: Optional[str] = None
    """Name of column with city names."""
    district_label: Optional[str] = None
    """Name of column with district names."""
    street_label: Optional[str] = None
    """Name of column with street names."""
    house_number_label: Optional[str] = None
    """Name of column with house numbers."""
    flat_label: Optional[str] = None
    """Name of column with flat numbers."""
    house_name_label: Optional[str] = None
    """Name of column with house names."""
    full_address_label: Optional[str] = None
    """Name of column with full addresses."""


@dataclass(frozen=True)
class PersonLabels:
    """Column labels for the Person annotation.

    See also:

        :class:`~synthesized.metadata.value.Person` : Annotate a dataset with address information.
    """
    title_label: Optional[str] = None
    """Name of column with title (e.g Mr, Mrs)."""
    gender_label: Optional[str] = None
    """Name of column with genders (e.g Male, Female)."""
    name_label: Optional[str] = None
    """Name of column with full names."""
    fullname_label: Optional[str] = None
    """Name of column with full names."""
    firstname_label: Optional[str] = None
    """Name of column with first names."""
    lastname_label: Optional[str] = None
    """Name of column with last names."""
    email_label: Optional[str] = None
    """Name of column with email addresses."""
    username_label: Optional[str] = None
    """Name of column with usernames names."""
    password_label: Optional[str] = None
    """Name of column with passwords"""
    mobile_number_label: Optional[str] = None
    """Name of column with mobile telephone numbers."""
    home_number_label: Optional[str] = None
    """Name of column with house telephone numbers."""
    work_number_label: Optional[str] = None
    """Name of column with work telephone numbers."""


@dataclass(frozen=True)
class BankLabels:
    """Column labels for the Bank annotation.

    See also:

        :class:`~synthesized.metadata.value.Bank` : Annotate a dataset with bank information.
    """
    bic_label: Optional[str] = None
    """Name of column with work telephone numbers."""
    sort_code_label: Optional[str] = None
    """Name of column with sort codes in format XX-XX-XX."""
    account_label: Optional[str] = None
    """Name of column with bank account number."""


@dataclass
class MetaFactoryConfig:
    """
    Attributes:
        min_nim_unique: if number of unique values in pd.Series
            is below this a Categorical meta is returned.
    """
    parsing_nan_fraction_threshold: float = 0.25

    @property
    def meta_factory_config(self):
        return MetaFactoryConfig(**{f.name: getattr(self, f.name)
                                    for f in fields(MetaFactoryConfig)})


# Model Config Classes ----------------------------------------

@dataclass
class GenderModelConfig:
    gender_female_regex: str = r'^(f|female)$'
    gender_male_regex: str = r'^(m|male)$'
    gender_non_binary_regex: str = r'^(n|non\Wbinary|u|undefined|NA)$'
    title_female_regex: str = r'^(ms|mrs|miss|dr)\.?$'
    title_male_regex: str = r'^(mr|dr)\.?$'
    title_non_binary_regex: str = r'^(ind|per|m|mx)\.?$'
    genders: Tuple[str, ...] = ('F', 'M')

    def gender_model_config(self):
        return GenderModelConfig(
            **{f.name: getattr(self, f.name) for f in fields(GenderModelConfig)}
        )


@dataclass
class PersonModelConfig(GenderModelConfig):
    person_locale: str = 'en'
    """Locale for name synthesis."""
    dict_cache_size: int = 10000
    """Size of cache for name generation."""
    mobile_number_format: str = '07xxxxxxxx'
    """Format of mobile telephone number."""
    home_number_format: str = '02xxxxxxxx'
    """Format of home telephone number."""
    work_number_format: str = '07xxxxxxxx'
    """Format of work telephone number."""
    pwd_length: Tuple[int, int] = (8, 12)  # (min, max)
    """Length of password."""

    def __post_init__(self):
        try:
            provider = importlib.import_module(f"faker.providers.person.{self.person_locale}")
            _ = provider.Provider.first_names
            _ = provider.Provider.last_names
        except ModuleNotFoundError:
            raise ValueError(f"Given locale '{self.person_locale}' not supported.")

    @property
    def person_model_config(self):
        return PersonModelConfig(**{f.name: getattr(self, f.name) for f in fields(PersonModelConfig)})


@dataclass
class PostcodeModelConfig:
    postcode_regex: str = r'([A-Za-z]{1,2})([0-9]+[A-Za-z]?)( *[0-9]+[A-Za-z]{2})'
    """Regular expression for postcode synthesis."""
    postcode_level: int = 0
    """Level of the postcode to learn."""

    @property
    def postcode_model_config(self):
        return PostcodeModelConfig(**{f.name: getattr(self, f.name) for f in fields(PostcodeModelConfig)})


@dataclass
class AddressModelConfig(PostcodeModelConfig):
    address_locale: str = 'en_GB'
    """Locale for address synthesis."""
    addresses_file: Optional[str] = '~/.synthesized/addresses.jsonl.gz'
    """Path to file with pre-generated addresses."""
    learn_postcodes: bool = False
    """Whether to learn postcodes from original data, or synthesis new examples."""

    def __post_init__(self):
        if self.address_locale != 'en_GB':
            # TODO: Handle postcodes from other coutries.
            raise ValueError(f"Given locale '{self.address_locale}' not supported, only 'en_GB' is supported.")

    @property
    def address_model_config(self):
        return AddressModelConfig(**{f.name: getattr(self, f.name) for f in fields(AddressModelConfig)})


@dataclass
class ModelBuilderConfig(AddressModelConfig, PersonModelConfig):
    categorical_threshold_log_multiplier: float = 2.5
    min_num_unique: int = 10

    @property
    def model_builder_config(self):
        return ModelBuilderConfig(**{f.name: getattr(self, f.name) for f in fields(ModelBuilderConfig)})


# Transformer Config Classes ----------------------------------------
@dataclass
class QuantileTransformerConfig:
    n_quantiles: int = 1000
    distribution: str = 'normal'
    noise: Optional[float] = 1e-7

    @property
    def quantile_transformer_config(self) -> 'QuantileTransformerConfig':
        return QuantileTransformerConfig(
            **{f.name: getattr(self, f.name) for f in fields(QuantileTransformerConfig)}
        )


@dataclass
class DateTransformerConfig(QuantileTransformerConfig):
    unit: str = 'days'

    @property
    def date_transformer_config(self) -> 'DateTransformerConfig':
        return DateTransformerConfig(
            **{f.name: getattr(self, f.name) for f in fields(DateTransformerConfig)}
        )


@dataclass
class MetaTransformerConfig(DateTransformerConfig, QuantileTransformerConfig):
    pass

# Value Config Classes ----------------------------------------


@dataclass
class CategoricalConfig:
    categorical_weight: float = 3.5
    temperature: float = 1.0
    moving_average: bool = True

    @property
    def categorical_config(self):
        return CategoricalConfig(**{f.name: getattr(self, f.name) for f in fields(CategoricalConfig)})


@dataclass
class ContinuousConfig:
    continuous_weight: float = 5.0

    @property
    def continuous_config(self):
        return ContinuousConfig(**{f.name: getattr(self, f.name) for f in fields(ContinuousConfig)})


@dataclass
class DecomposedContinuousConfig(ContinuousConfig):
    low_freq_weight: float = 1.0
    high_freq_weight: float = 1.0

    @property
    def decomposed_continuous_config(self):
        return DecomposedContinuousConfig(
            **{f.name: getattr(self, f.name) for f in fields(DecomposedContinuousConfig)}
        )


@dataclass
class IdentifierConfig:
    capacity: int = 128

    @property
    def identifier_config(self):
        return IdentifierConfig(**{f.name: getattr(self, f.name) for f in fields(IdentifierConfig)})


@dataclass
class NanConfig:
    nan_weight: float = 1.0

    @property
    def nan_config(self):
        return NanConfig(**{f.name: getattr(self, f.name) for f in fields(NanConfig)})


@dataclass
class ValueFactoryConfig(CategoricalConfig, NanConfig, IdentifierConfig, DecomposedContinuousConfig):
    capacity: int = 128

    @property
    def value_factory_config(self):
        return ValueFactoryConfig(**{f.name: getattr(self, f.name) for f in fields(ValueFactoryConfig)})


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
        return LearningManagerConfig(**{f.name: getattr(self, f.name) for f in fields(LearningManagerConfig)})


@dataclass
class DifferentialPrivacyConfig:
    """
    epsilon: epsilon value from differential privacy definition.
    delta: delta value from differential privacy definition. Should be significantly smaller than 1/data_size. If None,
    then delta is assumed to be 1/(10*data_size).
    num_microbatches: number of microbatches on which average gradient is calculated for clipping. Must evenly divide
    the batch size. When num_microbatches == batch_size the optimal utility can be achieved, however this results
    in a decrease in performance due to the number of extra calculations required.
    noise_multiplier: Amount of noise sampled and added to gradients during training. More noise results in higher
    privacy but lower utility
    l2_norm_clip: Maximum L2 norm of each microbatch gradient.

    See https://github.com/tensorflow/privacy for further details."""
    epsilon: float = 1.0
    """Abort model training when this value of epsilon is reached."""
    delta: Optional[float] = None
    """The delta in (epsilon, delta)-differential privacy. By default, set to 1/(10*len(data))."""
    noise_multiplier: float = 1.0
    """Ratio that determines amount of noise added to each sample during training"""
    num_microbatches: int = 1
    """Number of microbatches on which average gradient is calculated for clipping. Must evenly divide
    the batch size.

    When num_microbatches == batch_size the optimal utility can be achieved, however this results
    in a decrease in performance due to the number of extra calculations required."""
    l2_norm_clip: float = 1.0
    """Maximum L2 norm of each microbatch gradient."""

    @property
    def privacy_config(self):
        return DifferentialPrivacyConfig(**{f.name: getattr(self, f.name) for f in fields(DifferentialPrivacyConfig)})


@dataclass
class EngineConfig(DifferentialPrivacyConfig):
    """
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
    beta: beta.
    weight_decay: Weight decay.
    differential_privacy: Whether to enable differential privacy during training.
    """
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
    # Losses
    beta: float = 1.0
    weight_decay: float = 1e-3
    # Differential privacy
    differential_privacy: bool = False
    """Enable differential privacy."""

    @property
    def engine_config(self):
        return EngineConfig(**{f.name: getattr(self, f.name) for f in fields(EngineConfig)})

# Synthesizer Config Classes ----------------------------------------


@dataclass
class HighDimConfig(EngineConfig, ValueFactoryConfig, LearningManagerConfig, ModelBuilderConfig):
    """Configuration for :class:`synthesized.HighDimSynthesizer`."""
    distribution: str = 'normal'
    batch_size: int = 64
    increase_batch_size_every: Optional[int] = 500
    max_batch_size: Optional[int] = 1024
    synthesis_batch_size: Optional[int] = 16384
    learning_manager: bool = True
    """Control learning with the LearningManager."""


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
