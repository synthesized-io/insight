from typing import Dict, List, Union
from enum import Enum

import pandas as pd

from .data_frame_meta import DataFrameMeta
from ..config import MetaExtractorConfig
from ..config import AddressParams
from ..config import BankParams
from ..config import CompoundAddressParams
from ..config import PersonParams
from ..metadata import MetaExtractor as _MetaExtractor


class TypeOverride(Enum):
    ID = 'ID'
    DATE = 'DATE'
    CATEGORICAL = 'CATEGORICAL'
    CONTINUOUS = 'CONTINUOUS'
    ENUMERATION = 'ENUMERATION'


class MetaExtractor:
    def __init__(self, config: MetaExtractorConfig = MetaExtractorConfig()):
        self._meta_extractor = _MetaExtractor(config=config)

    @classmethod
    def extract(
            cls, df: pd.DataFrame, config: MetaExtractorConfig = MetaExtractorConfig(),
            id_index: str = None, time_index: str = None,
            column_aliases: Dict[str, str] = None, associations: Dict[str, List[str]] = None,
            type_overrides: Dict[str, TypeOverride] = None,
            find_rules: Union[str, List[str]] = None, produce_nans_for: List[str] = None,
            produce_infs_for: List[str] = None,
            address_params: AddressParams = None, bank_params: BankParams = None,
            compound_address_params: CompoundAddressParams = None,
            person_params: PersonParams = None
    ) -> DataFrameMeta:
        pass

    def extract_dataframe_meta(
            self, df: pd.DataFrame, id_index: str = None, time_index: str = None,
            column_aliases: Dict[str, str] = None, associations: Dict[str, List[str]] = None,
            type_overrides: Dict[str, TypeOverride] = None,
            find_rules: Union[str, List[str]] = None, produce_nans_for: List[str] = None,
            produce_infs_for: List[str] = None,
            address_params: AddressParams = None, bank_params: BankParams = None,
            compound_address_params: CompoundAddressParams = None,
            person_params: PersonParams = None
    ) -> DataFrameMeta:
        pass
