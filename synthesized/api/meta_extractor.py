# type: ignore
from typing import List, Union

import pandas as pd

from .annotations import Address, Bank, Person
from .data_frame_meta import DataFrameMeta
from ..config import MetaFactoryConfig
from ..metadata_new import MetaExtractor as _MetaExtractor

Annotations = Union[Address, Bank, Person]


class MetaExtractor:
    """Extracts metadata from the columns of a tabular dataset. Used to produce DataFrameMeta objects."""
    def __init__(self, config: MetaFactoryConfig = MetaFactoryConfig()):
        """Initialises a MetaExtractor instance with a provided configuration.

        Args:
            config: A MetaExtractor configuration object.
        """
        self._meta_extractor = _MetaExtractor(config=config)

    @classmethod
    def extract(
            cls, df: pd.DataFrame, config: MetaFactoryConfig = MetaFactoryConfig(),
            annotations: List[Annotations] = None
    ) -> DataFrameMeta:
        """Extracts the DataFrame metadata with the provided configuration options.

        Args:
            df: A pandas DataFrame containing the dataset to be extracted from.
            config: Configuration for the MetaExtractor.
            annotations: list of annotation objects for the dataframe

        Returns:
            The extracted DataFrameMeta.
        """
        df_meta = DataFrameMeta()
        _annotations = None
        if annotations is not None:
            _annotations = [annotation._annotation for annotation in annotations]

        df_meta._df_meta = _MetaExtractor.extract(
            df=df, config=config, annotations=_annotations
        )
        return df_meta

    def extract_dataframe_meta(
            self, df: pd.DataFrame, config: MetaFactoryConfig = MetaFactoryConfig(),
            annotations: List[Annotations] = None
    ) -> DataFrameMeta:
        """Extracts the DataFrame metadata with the instance's configuration options.

        This is the instance method equivalent of the class method, MetaExtractor.extract.

        Args:
            df: A pandas DataFrame containing the dataset to be extracted from.
            config: Configuration for the MetaExtractor.
            annotations: list of annotation objects for the dataframe

        Returns:
            The extracted DataFrameMeta.
        """
        df_meta = DataFrameMeta()
        _annotations = [annotation._annotation for annotation in annotations]
        df_meta._df_meta = _MetaExtractor.extract(
            df=df, config=config, annotations=_annotations
        )
        return df_meta
