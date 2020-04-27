from .metrics import StandardDeviation  # noqa: F401
from .metrics import KendellTauCorrelation  # noqa: F401
from .metrics import SpearmanRhoCorrelation  # noqa: F401
from .metrics import CramersV  # noqa: F401
from .metrics import KolmogorovSmirnovDistance  # noqa: F401
from .metrics import EarthMoversDistance  # noqa: F401

from .metrics_base import ColumnMetric  # noqa: F401
from .metrics_base import TwoColumnMetric  # noqa: F401
from .metrics_base import DataFrameMetric  # noqa: F401
from .metrics_base import ColumnComparison  # noqa: F401
from .metrics_base import TwoColumnComparison  # noqa: F401
from .metrics_base import DataFrameComparison  # noqa: F401
from .metrics_base import _Metric  # noqa: F401

# As we dynamically create Min/Max/Avg/Diff metrics, we must register the classes globally
for metric in _Metric.ALL:
    globals()[metric.__name__] = metric


__all__ = ['ColumnMetric', 'TwoColumnMetric', 'DataFrameMetric', 'ColumnComparison', 'TwoColumnComparison',
           'DataFrameComparison'] + [metric.__name__ for metric in _Metric.ALL]
