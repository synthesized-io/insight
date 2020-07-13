from typing import Any, Dict, List
from base64 import b64encode, b64decode

import pickle
import pandas as pd


class ValueMeta:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.__class__.__name__[:-4].lower() + "_meta"

    def specification(self):
        return dict(name=self.name)

    def columns(self) -> List[str]:
        """External columns which are covered by this value.

        Returns:
            Columns covered by this value.

        """
        return [self.name]

    def learned_input_columns(self) -> List[str]:
        """Internal input columns for a generative model.

        Returns:
            Learned input columns.

        """
        return [self.name]

    def learned_output_columns(self) -> List[str]:
        """Internal output columns for a generative model.

        Returns:
            Learned output columns.

        """
        return [self.name]

    def extract(self, df: pd.DataFrame) -> None:
        """Extracts configuration parameters from a representative data frame.

        Overwriting implementations should call super().extract(df=df) as first step.

        Args:
            df: Representative data frame.

        """
        assert all(name in df.columns for name in self.columns())

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-processes a data frame to prepare it as input for a generative model. This may
        include adding or removing columns in case of `learned_input_columns()` differing from
        `columns()`.

        Important: this function modifies the given data frame.

        Overwriting implementations should call super().preprocess(df=df) as last step.

        Args:
            df: Data frame to be pre-processed.

        Returns:
            Pre-processed data frame.

        """
        assert all(name in df.columns for name in self.learned_input_columns())
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes a data frame, usually the output of a generative model. Post-processing
        basically reverses the pre-processing procedure. This may include re-introducing columns in
        case of `learned_output_columns()` differing from `columns()`.

        Important: this function modifies the given data frame.

        Overwriting implementations should call super().postprocess(df=df) as first step.

        Args:
            df: Data frame to be post-processed.

        Returns:
            Post-processed data frame.

        """
        assert all(name in df.columns for name in self.learned_output_columns())
        return df

    def get_variables(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            pickle=b64encode(pickle.dumps(self)).decode('utf-8')
        )

    @staticmethod
    def set_variables(variables: Dict[str, Any]):
        return pickle.loads(b64decode(variables['pickle'].encode('utf-8')))
