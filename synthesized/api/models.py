from .data_frame_meta import DataFrameMeta
from ..metadata_new import Affine as _Affine
from ..metadata_new import Nominal as _Nominal
from ..model.models import Histogram as _Histogram
from ..model.models import KernelDensityEstimate as _KernelDensityEstimate


class DiscreteModel:
    """
    An object given to the HighDimSynthesizer to specify that a column should be treated as discrete.
    """
    def __init__(self, name: str, df_meta: DataFrameMeta):
        """
        Args
            name: column name in the dataframe
            df_meta: dataframe meta
        """
        assert df_meta._df_meta is not None, "underlying df_meta.df_meta is not defined"
        self._meta = df_meta._df_meta[name]
        if isinstance(self._meta, _Nominal):
            self._model = _Histogram(self._meta)
        else:
            raise ValueError("Column specified cannot be treated as a discrete model")


class ContinuousModel:
    """
    An object given to the HighDimSynthesizer to specify that a column should be treated as continuous.
    """
    def __init__(self, name: str, df_meta: DataFrameMeta):
        """
        Args
            name: column name in the dataframe
            df_meta: dataframe meta
        """
        assert df_meta._df_meta is not None, "underlying df_meta.df_meta is not defined"
        self._meta = df_meta._df_meta[name]
        if isinstance(self._meta, _Affine):
            self._model = _KernelDensityEstimate(self._meta)
        else:
            raise ValueError("Column specified cannot be treated as continuous model")
