"""TODO: deprecated under the current new metadata, needs updating """
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from ..metadata.data_frame_meta import DataFrameMeta

logger = logging.getLogger(__name__)


class TestingDataGenerator:
    def __init__(self, df_meta: DataFrameMeta):
        self.df_meta = df_meta

    @classmethod
    def from_dict(cls, value_kwargs: Optional[List[Dict[str, Any]]],
                  groups_kwargs: Optional[List[Dict[str, Any]]]) -> 'TestingDataGenerator':
        raise NotImplementedError

    @classmethod
    def from_yaml(cls, config_file_name: str):

        logger.info("Reading configuration file.")
        with open(config_file_name, 'r') as fp:
            kwargs = yaml.safe_load(fp)

        values = kwargs.get("values", None)
        groups = kwargs.get("groups", {})
        generator = cls.from_dict(values, groups)

        return generator

    def synthesize(self, num_rows: int) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def validate_arguments(args: List[str], kwargs: Dict[str, Any]) -> List:
        raise NotImplementedError
