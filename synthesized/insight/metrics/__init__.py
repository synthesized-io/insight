from itertools import chain

from .metrics import (CategoricalLogisticR2, CramersV, EarthMoversDistance, KendellTauCorrelation,
                      KolmogorovSmirnovDistance, Mean, SpearmanRhoCorrelation, StandardDeviation)
from .metrics_base import ChainColumnVector  # noqa: F401
from .metrics_base import ColumnVector  # noqa: F401
from .metrics_base import DataFrameMatrix  # noqa: F401
from .metrics_base import DataFrameVector  # noqa: F401
from .metrics_base import RollingColumnMetricVector  # noqa: F401
from .metrics_base import TwoDataFrameMatrix  # noqa: F401
from .metrics_base import TwoDataFrameVector  # noqa: F401
from .metrics_base import (ClassificationMetric, ClassificationPlotMetric, ColumnComparisonVector, ColumnMetric,
                           ColumnMetricVector, DataFrameMetric, DiffColumnMetricAdapter, DiffMetricMatrix,
                           ModellingMetric, RegressionMetric, TwoColumnMetric, TwoColumnMetricMatrix,
                           TwoDataFrameMetric, _Matrix, _Metric, _Vector)
from .modelling_metrics import (ROC_AUC, Accuracy, ConfusionMatrix, F1Score, MeanAbsoluteError, MeanSquaredError,
                                PR_Curve, Precision, R2_Score, Recall, ROC_Curve, predictive_modelling_comparison,
                                predictive_modelling_score)
from .vectors import DiffVector, FractionalDiffVector

_CORE_METRICS = [
    Mean, StandardDeviation, KendellTauCorrelation, SpearmanRhoCorrelation, CramersV, KolmogorovSmirnovDistance,
    EarthMoversDistance, CategoricalLogisticR2
]

# As we dynamically create Min/Max/Avg/Diff metrics, we must register the classes globally
for core_metric in _CORE_METRICS:
    core_metric()

for metric_name, metric in _Metric.ALL.items():
    globals()[metric_name] = metric
for vector_name, vector in _Vector.ALL.items():
    globals()[vector_name] = vector
for matrix_name, matrix in _Matrix.ALL.items():
    globals()[matrix_name] = matrix


ALL = _Metric.ALL
COLUMN_METRICS = ColumnMetric.ALL
TWO_COLUMN_METRICS = TwoColumnMetric.ALL
DATA_FRAME_METRICS = DataFrameMetric.ALL
TWO_DATA_FRAME_METRICS = TwoDataFrameMetric.ALL
VECTORS = _Vector.ALL
MATRICES = _Matrix.ALL

# ColumnMetrics
# -----------------------------------------------------------------------------
mean = Mean()
standard_deviation = StandardDeviation()

# TwoColumnMetrics
# -----------------------------------------------------------------------------
kendell_tau_correlation = KendellTauCorrelation()
spearman_rho_correlation = SpearmanRhoCorrelation()
cramers_v = CramersV()
categorical_logistic_correlation = CategoricalLogisticR2()

diff_standard_deviation = DiffColumnMetricAdapter(standard_deviation)
kolmogorov_smirnov_distance = KolmogorovSmirnovDistance()
earth_movers_distance = EarthMoversDistance()

# DataFrameMetrics
# -----------------------------------------------------------------------------
# predictive_modelling_score = PredictiveModellingScore()

# TwoDataFrameMetrics
# -----------------------------------------------------------------------------
# predictive_modelling_comparison = PredictiveModellingComparison()

# DataFrameMetricVectors
# -----------------------------------------------------------------------------
diff_vector = DiffVector()
fractional_diff_vector = FractionalDiffVector()

# DataFrameMetricVectors
# -----------------------------------------------------------------------------
standard_deviation_vector = ColumnMetricVector(standard_deviation)

# TwoDataFrameVectors
# -----------------------------------------------------------------------------
kolmogorov_smirnov_distance_vector = ColumnComparisonVector(kolmogorov_smirnov_distance)
earth_movers_distance_vector = ColumnComparisonVector(earth_movers_distance)

# DataFrameMetricMatrices
# -----------------------------------------------------------------------------
kendell_tau_correlation_matrix = TwoColumnMetricMatrix(kendell_tau_correlation)
spearman_rho_correlation_matrix = TwoColumnMetricMatrix(spearman_rho_correlation)
cramers_v_matrix = TwoColumnMetricMatrix(cramers_v)
categorical_logistic_correlation_matrix = TwoColumnMetricMatrix(categorical_logistic_correlation)

# DataFrameComparisonMatrices
# -----------------------------------------------------------------------------
diff_kendell_tau_correlation_matrix = DiffMetricMatrix(kendell_tau_correlation_matrix)
diff_spearman_rho_correlation_matrix = DiffMetricMatrix(spearman_rho_correlation_matrix)
diff_cramers_v_matrix = DiffMetricMatrix(cramers_v_matrix)
diff_categorical_logistic_correlation_matrix = DiffMetricMatrix(categorical_logistic_correlation_matrix)

# Modelling Metrics
# -----------------------------------------------------------------------------
# Classification
accuracy = Accuracy()
precision = Precision()
recall = Recall()
f1_score = F1Score()
roc_auc = ROC_AUC()

# Classification - Plot
roc_curve = ROC_Curve()
pr_curve = PR_Curve()
confusion_matrix = ConfusionMatrix()

# Regression
mean_absolute_error = MeanAbsoluteError()
mean_squared_error = MeanSquaredError()
r2_score = R2_Score()


__all__ = [
    'ALL', 'COLUMN_METRICS', 'TWO_COLUMN_METRICS', 'DATA_FRAME_METRICS', 'TWO_DATA_FRAME_METRICS', 'ColumnMetric',
    'TwoColumnMetric', 'DataFrameMetric', 'TwoDataFrameMetric', 'DataFrameVector',
    'TwoDataFrameVector', 'DataFrameMatrix', 'TwoDataFrameMatrix', 'DiffMetricMatrix', 'RollingColumnMetricVector',
    'ModellingMetric', 'ClassificationMetric', 'RegressionMetric', 'ClassificationPlotMetric',
    'predictive_modelling_score', 'predictive_modelling_comparison'
]
__all__.extend([metric_name for metric_name in chain(_Metric.ALL.keys(), _Vector.ALL.keys(), _Matrix.ALL.keys())])
