from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

# Meta Config Classes ----------------------------------------


@dataclass(repr=True)
class AddressRecord:
    postcode: Optional[str] = None
    county: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    street: Optional[str] = None
    house_number: Optional[str] = None
    flat: Optional[str] = None
    house_name: Optional[str] = None
    full_address: str = field(init=False, repr=False)

    def __post_init__(self):
        for f in fields(self):
            if f.name != 'full_address':
                field_val = getattr(self, f.name)
                setattr(self, f.name, field_val.replace("'", "") if field_val is not None else None)
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


@dataclass
class PostcodeModelConfig:
    postcode_regex: str = r'[A-Za-z]{1,2}[0-9]+[A-Za-z]? *[0-9]+[A-Za-z]{2}'

    @property
    def postcode_model_config(self):
        return PostcodeModelConfig(**{f.name: getattr(self, f.name) for f in fields(PostcodeModelConfig)})


@dataclass
class AddressModelConfig(PostcodeModelConfig):
    locale: str = 'en_GB'
    postcode_level: int = 0
    addresses_file: Optional[str] = '~/.synthesized/addresses.jsonl.gz'
    learn_postcodes: bool = False

    @property
    def address_model_config(self):
        return AddressModelConfig(**{f.name: getattr(self, f.name) for f in fields(AddressModelConfig)})


@dataclass(frozen=True)
class AddressLabels:
    postcode_label: Optional[str] = None
    county_label: Optional[str] = None
    city_label: Optional[str] = None
    district_label: Optional[str] = None
    street_label: Optional[str] = None
    house_number_label: Optional[str] = None
    flat_label: Optional[str] = None
    house_name_label: Optional[str] = None
    full_address_label: Optional[str] = None


@dataclass(frozen=True)
class PersonLabels:
    title_label: Optional[str] = None
    gender_label: Optional[str] = None
    name_label: Optional[str] = None
    fullname_label: Optional[str] = None
    firstname_label: Optional[str] = None
    lastname_label: Optional[str] = None
    email_label: Optional[str] = None
    username_label: Optional[str] = None
    password_label: Optional[str] = None
    mobile_number_label: Optional[str] = None
    home_number_label: Optional[str] = None
    work_number_label: Optional[str] = None


@dataclass(frozen=True)
class BankLabels:
    bic_label: Optional[str] = None
    sort_code_label: Optional[str] = None
    account_label: Optional[str] = None


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
    locale: str = 'en'
    dict_cache_size: int = 10000
    mobile_number_format: str = '07xxxxxxxx'
    home_number_format: str = '02xxxxxxxx'
    work_number_format: str = '07xxxxxxxx'
    pwd_length: Tuple[int, int] = (8, 12)  # (min, max)

    @property
    def person_model_config(self):
        return PersonModelConfig(**{f.name: getattr(self, f.name) for f in fields(PersonModelConfig)})


@dataclass
class FormattedStringParams:
    formatted_string_label: Optional[List[str]] = None


@dataclass
class FormattedStringMetaConfig:
    label_to_regex: Optional[Dict[str, str]] = None

    @property
    def formatted_string_meta_config(self):
        return FormattedStringMetaConfig(**{f.name: getattr(self, f.name)
                                            for f in fields(FormattedStringMetaConfig)})


@dataclass
class MetaFactoryConfig:
    """
    Attributes:
        categorical_threshold_log_multiplier: if number of unique values
            in a pd.Series is below this value a Categorical meta is returned.
        min_nim_unique: if number of unique values in pd.Series
            is below this a Categorical meta is returned.
        acceptable_nan_frac: when interpreting a series of type 'O',
            data is cast to numeric and non numeric types are cast to
            NaNs. If the frequency of NaNs is below this threshold, and
            Categorcial meta has not been inferred, then Float or Integer meta
            is returned.
    """
    categorical_threshold_log_multiplier: float = 2.5
    parsing_nan_fraction_threshold: float = 0.25
    min_num_unique: int = 10


@dataclass
class MetaExtractorConfig(MetaFactoryConfig, AddressModelConfig, PersonModelConfig, FormattedStringMetaConfig):

    @property
    def meta_extractor_config(self):
        return MetaExtractorConfig(**{f.name: getattr(self, f.name) for f in fields(MetaExtractorConfig)})


@dataclass
class ModelBuilderConfig(PersonModelConfig):
    categorical_threshold_log_multiplier: float = 2.5
    min_num_unique: int = 10


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
class EngineConfig:
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

    @property
    def engine_config(self):
        return EngineConfig(**{f.name: getattr(self, f.name) for f in fields(EngineConfig)})

# Synthesizer Config Classes ----------------------------------------


@dataclass
class HighDimConfig(EngineConfig, ValueFactoryConfig, LearningManagerConfig):
    """
    distribution: Distribution type: "normal".
    batch_size: Batch size.
    learning_manager: Whether to use LearningManager.
    """
    distribution: str = 'normal'
    batch_size: int = 64
    increase_batch_size_every: Optional[int] = 500
    max_batch_size: Optional[int] = 1024
    synthesis_batch_size: Optional[int] = 16384
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
