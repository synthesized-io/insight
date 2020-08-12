from typing import Dict, List, Union

import pandas as pd

from .data_frame_meta import DataFrameMeta
from ..config import MetaExtractorConfig
from ..config import AddressParams
from ..config import BankParams
from ..config import CompoundAddressParams
from ..config import PersonParams
from ..metadata import MetaExtractor as _MetaExtractor, TypeOverride


class MetaExtractor:
    def __init__(self, config: MetaExtractorConfig = MetaExtractorConfig()):
        self._meta_extractor = _MetaExtractor(config=config)

    @classmethod
    def extract(
            cls,
            df: pd.DataFrame,
            config: MetaExtractorConfig = MetaExtractorConfig(),
            id_index: str = None,
            time_index: str = None,
            column_aliases: Dict[str, str] = None,
            associations: Dict[str, List[str]] = None,
            type_overrides: Dict[str, TypeOverride] = None,
            find_rules: Union[str, List[str]] = None,
            produce_nans_for: List[str] = None,
            produce_infs_for: List[str] = None,
            address_params: AddressParams = None,
            bank_params: BankParams = None,
            compound_address_params: CompoundAddressParams = None,
            person_params: PersonParams = None,
    ) -> DataFrameMeta:

        df_meta = DataFrameMeta()
        df_meta._df_meta = _MetaExtractor.extract(
            df=df,
            config=config,
            id_index=id_index,
            time_index=time_index,
            column_aliases=column_aliases,
            associations=associations,
            type_overrides=type_overrides,
            find_rules=find_rules,
            produce_nans_for=produce_nans_for,
            produce_infs_for=produce_infs_for,
            address_params=address_params,
            bank_params=bank_params,
            compound_address_params=compound_address_params,
            person_params=person_params,
        )
        return df_meta

    def extract_dataframe_meta(
            self,
            df: pd.DataFrame,
            id_index: str = None,
            time_index: str = None,
            column_aliases: Dict[str, str] = None,
            associations: Dict[str, List[str]] = None,
            type_overrides: Dict[str, TypeOverride] = None,
            find_rules: Union[str, List[str]] = None,
            produce_nans_for: List[str] = None,
            produce_infs_for: List[str] = None,
            address_params: AddressParams = None,
            bank_params: BankParams = None,
            compound_address_params: CompoundAddressParams = None,
            person_params: PersonParams = None,
    ) -> DataFrameMeta:

        df_meta = DataFrameMeta()
        df_meta._df_meta = self._meta_extractor.extract(
            df=df,
            id_index=id_index,
            time_index=time_index,
            column_aliases=column_aliases,
            associations=associations,
            type_overrides=type_overrides,
            find_rules=find_rules,
            produce_nans_for=produce_nans_for,
            produce_infs_for=produce_infs_for,
            address_params=address_params,
            bank_params=bank_params,
            compound_address_params=compound_address_params,
            person_params=person_params,
        )
        return df_meta
