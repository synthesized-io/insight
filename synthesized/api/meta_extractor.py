from typing import Dict, List

import pandas as pd

from ..config import AddressParams, BankParams, MetaExtractorConfig, PersonParams
from ..metadata import MetaExtractor as _MetaExtractor
from ..metadata import TypeOverride
from .data_frame_meta import DataFrameMeta


class MetaExtractor:
    """Extracts metadata from the columns of a tabular dataset. Used to produce DataFrameMeta objects."""
    def __init__(self, config: MetaExtractorConfig = MetaExtractorConfig()):
        """Initialises a MetaExtractor instance with a provided configuration.

        Args:
            config: A MetaExtractor configuration object.
        """
        self._meta_extractor = _MetaExtractor(config=config)

    @classmethod
    def extract(
            cls, df: pd.DataFrame, config: MetaExtractorConfig = MetaExtractorConfig(),
            column_aliases: Dict[str, str] = None, associations: Dict[str, List[str]] = None,
            type_overrides: Dict[str, TypeOverride] = None,
            address_params: AddressParams = None, bank_params: BankParams = None, person_params: PersonParams = None
    ) -> DataFrameMeta:
        """Extracts the DataFrame metadata with the provided configuration options.

        Args:
            df: A pandas DataFrame containing the dataset to be extracted from.
            config: Configuration for the MetaExtractor.
            column_aliases: Optional dictionary mapping pairs of column aliases.
            associations: Optional dictionary assigning strict associations between categories.
            type_overrides: Optional dictionary mapping column names to desired TypeOverrides.
            address_params: Parameters for Address annotations.
            bank_params: Parameters for Bank Account annotations.
            person_params: Parameters for Person annotations

        Returns:
            The extracted DataFrameMeta.
        """
        df_meta = DataFrameMeta()
        df_meta._df_meta = _MetaExtractor.extract(
            df=df, config=config, id_index=None, time_index=None, column_aliases=column_aliases,
            associations=associations, type_overrides=type_overrides, find_rules=None,
            address_params=address_params,
            bank_params=bank_params, person_params=person_params,
        )
        return df_meta

    def extract_dataframe_meta(
            self, df: pd.DataFrame, column_aliases: Dict[str, str] = None,
            associations: Dict[str, List[str]] = None, type_overrides: Dict[str, TypeOverride] = None,
            address_params: AddressParams = None, bank_params: BankParams = None, person_params: PersonParams = None
    ) -> DataFrameMeta:
        """Extracts the DataFrame metadata with the instance's configuration options.

        This is the instance method equivalent of the class method, MetaExtractor.extract.

        Args:
            df: A pandas DataFrame containing the dataset to be extracted from.
            column_aliases: Optional dictionary mapping pairs of column aliases.
            associations: Optional dictionary assigning strict associations between categories.
            type_overrides: Optional dictionary mapping column names to desired TypeOverrides.
            address_params: Parameters for Address annotations.
            bank_params: Parameters for Bank Account annotations.
            person_params: Parameters for Person annotations

        Returns:
            The extracted DataFrameMeta.
        """
        df_meta = DataFrameMeta()
        df_meta._df_meta = self._meta_extractor.extract(
            df=df, id_index=None, time_index=None, column_aliases=column_aliases, associations=associations,
            type_overrides=type_overrides, find_rules=None, address_params=address_params, bank_params=bank_params,
            person_params=person_params
        )
        return df_meta
