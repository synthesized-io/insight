from .metrics import StandardDeviation
from .metrics import KendellTauCorrelation
from .metrics import SpearmanRhoCorrelation
from .metrics import CramersV
from .metrics import KolmogorovSmirnovDistance
from .metrics import EarthMoversDistance
from .metrics import PredictiveModellingScore
from .metrics import PredictiveModellingComparison

from .metrics_base import ColumnMetric
from .metrics_base import TwoColumnMetric
from .metrics_base import DataFrameMetric
from .metrics_base import ColumnComparison
from .metrics_base import TwoColumnComparison
from .metrics_base import DataFrameComparison
from .metrics_base import _Metric


_CORE_METRICS = [
    StandardDeviation, KendellTauCorrelation, SpearmanRhoCorrelation, CramersV, KolmogorovSmirnovDistance,
    EarthMoversDistance, PredictiveModellingScore, PredictiveModellingComparison
]

# As we dynamically create Min/Max/Avg/Diff metrics, we must register the classes globally
for metric in _CORE_METRICS:
    metric()

for metric_name, metric in _Metric.ALL.items():
    globals()[metric_name] = metric


ALL = _Metric.ALL
COLUMN_METRICS = ColumnMetric.ALL
TWO_COLUMN_METRICS = TwoColumnMetric.ALL
DATA_FRAME_METRICS = DataFrameMetric.ALL
COLUMN_COMPARISONS = ColumnComparison.ALL
TWO_COLUMN_COMPARISONS = TwoColumnComparison.ALL
DATA_FRAME_COMPARISONS = DataFrameComparison.ALL


__all__ = [
        'ALL', 'COLUMN_METRICS', 'TWO_COLUMN_METRICS', 'DATA_FRAME_METRICS', 'COLUMN_COMPARISONS',
        'TWO_COLUMN_COMPARISONS', 'DATA_FRAME_COMPARISONS' 'ColumnMetric', 'TwoColumnMetric', 'DataFrameMetric',
        'ColumnComparison', 'TwoColumnComparison', 'DataFrameComparison'
] + [metric_name for metric_name in _Metric.ALL.keys()]
