from .metrics import Mean
from .metrics import StandardDeviation
from .metrics import KendellTauCorrelation
from .metrics import SpearmanRhoCorrelation
from .metrics import CramersV
from .metrics import KolmogorovSmirnovDistance
from .metrics import EarthMoversDistance
from .metrics import CategoricalLogisticR2

from .modelling_metrics import PredictiveModellingScore
from .modelling_metrics import PredictiveModellingComparison
from .modelling_metrics import Accuracy
from .modelling_metrics import Precision
from .modelling_metrics import Recall
from .modelling_metrics import F1Score
from .modelling_metrics import ROC_AUC
from .modelling_metrics import ROC_Curve
from .modelling_metrics import PR_Curve
from .modelling_metrics import ConfusionMatrix
from .modelling_metrics import MeanAbsoluteError
from .modelling_metrics import MeanSquaredError
from .modelling_metrics import R2_Score

from .vectors import DiffVector
from .vectors import FractionalDiffVector

from .metrics_base import _Metric
from .metrics_base import ColumnMetric
from .metrics_base import TwoColumnMetric
from .metrics_base import DataFrameMetric
from .metrics_base import ColumnComparison
from .metrics_base import TwoColumnComparison
from .metrics_base import DataFrameComparison

from .metrics_base import AggregateColumnMetricAdapter
from .metrics_base import AggregateTwoColumnMetricAdapter
from .metrics_base import AggregateColumnComparisonAdapter
from .metrics_base import AggregateTwoColumnComparisonAdapter
from .metrics_base import DiffColumnMetricAdapter
from .metrics_base import DiffTwoColumnMetricAdapter
from .metrics_base import DiffDataFrameMetricAdapter

from .metrics_base import _Vector
from .metrics_base import ColumnVector  # noqa: F401
from .metrics_base import DataFrameVector  # noqa: F401
from .metrics_base import DataFrameComparisonVector  # noqa: F401
from .metrics_base import ColumnMetricVector
from .metrics_base import ColumnComparisonVector
from .metrics_base import RollingColumnMetricVector  # noqa: F401
from .metrics_base import ChainColumnVector  # noqa: F401

from .metrics_base import _Matrix
from .metrics_base import DataFrameMatrix  # noqa: F401
from .metrics_base import DataFrameComparisonMatrix  # noqa: F401
from .metrics_base import TwoColumnMetricMatrix
from .metrics_base import TwoColumnComparisonMatrix

from .metrics_base import ModellingMetric
from .metrics_base import ClassificationMetric
from .metrics_base import RegressionMetric
from .metrics_base import ClassificationPlotMetric


_CORE_METRICS = [
    Mean, StandardDeviation, KendellTauCorrelation, SpearmanRhoCorrelation, CramersV, KolmogorovSmirnovDistance,
    EarthMoversDistance, CategoricalLogisticR2, PredictiveModellingScore, PredictiveModellingComparison
]

# As we dynamically create Min/Max/Avg/Diff metrics, we must register the classes globally
for metric in _CORE_METRICS:
    metric()

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
COLUMN_COMPARISONS = ColumnComparison.ALL
TWO_COLUMN_COMPARISONS = TwoColumnComparison.ALL
DATA_FRAME_COMPARISONS = DataFrameComparison.ALL
DATA_FRAME_VECTORS = _Vector.ALL
DATA_FRAME_MATRICES = _Matrix.ALL

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

# DataFrameMetrics
# -----------------------------------------------------------------------------
min_standard_deviation = AggregateColumnMetricAdapter(StandardDeviation(), summary_type='min')
max_standard_deviation = AggregateColumnMetricAdapter(StandardDeviation(), summary_type='max')
avg_standard_deviation = AggregateColumnMetricAdapter(StandardDeviation(), summary_type='avg')
min_kendell_tau_correlation = AggregateTwoColumnMetricAdapter(KendellTauCorrelation(), summary_type='min')
max_kendell_tau_correlation = AggregateTwoColumnMetricAdapter(KendellTauCorrelation(), summary_type='max')
avg_kendell_tau_correlation = AggregateTwoColumnMetricAdapter(KendellTauCorrelation(), summary_type='avg')
min_spearman_rho_correlation = AggregateTwoColumnMetricAdapter(SpearmanRhoCorrelation(), summary_type='min')
max_spearman_rho_correlation = AggregateTwoColumnMetricAdapter(SpearmanRhoCorrelation(), summary_type='max')
avg_spearman_rho_correlation = AggregateTwoColumnMetricAdapter(SpearmanRhoCorrelation(), summary_type='avg')
min_cramers_v = AggregateTwoColumnMetricAdapter(CramersV(), summary_type='min')
max_cramers_v = AggregateTwoColumnMetricAdapter(CramersV(), summary_type='max')
avg_cramers_v = AggregateTwoColumnMetricAdapter(CramersV(), summary_type='avg')
min_categorical_logistic_correlation = AggregateTwoColumnMetricAdapter(CategoricalLogisticR2(), summary_type='min')
max_categorical_logistic_correlation = AggregateTwoColumnMetricAdapter(CategoricalLogisticR2(), summary_type='max')
avg_categorical_logistic_correlation = AggregateTwoColumnMetricAdapter(CategoricalLogisticR2(), summary_type='avg')
predictive_modelling_score = PredictiveModellingScore()

# ColumnComparisons
# -----------------------------------------------------------------------------
diff_standard_deviation = DiffColumnMetricAdapter(StandardDeviation())
kolmogorov_smirnov_distance = KolmogorovSmirnovDistance()
earth_movers_distance = EarthMoversDistance()

# TwoColumnComparisons
# -----------------------------------------------------------------------------
diff_kendell_tau_correlation = DiffTwoColumnMetricAdapter(KendellTauCorrelation())
diff_spearman_rho_correlation = DiffTwoColumnMetricAdapter(SpearmanRhoCorrelation())
diff_cramers_v = DiffTwoColumnMetricAdapter(CramersV())
diff_categorical_logistic_correlation = DiffTwoColumnMetricAdapter(CategoricalLogisticR2())

# DataFrameComparisons
# -----------------------------------------------------------------------------
diff_min_standard_deviation = DiffDataFrameMetricAdapter(AggregateColumnMetricAdapter(StandardDeviation(), summary_type='min'))  # noqa: E501
diff_max_standard_deviation = DiffDataFrameMetricAdapter(AggregateColumnMetricAdapter(StandardDeviation(), summary_type='max'))  # noqa: E501
diff_avg_standard_deviation = DiffDataFrameMetricAdapter(AggregateColumnMetricAdapter(StandardDeviation(), summary_type='avg'))  # noqa: E501
min_diff_standard_deviation = AggregateColumnComparisonAdapter(DiffColumnMetricAdapter(StandardDeviation()), summary_type='min')  # noqa: E501
max_diff_standard_deviation = AggregateColumnComparisonAdapter(DiffColumnMetricAdapter(StandardDeviation()), summary_type='max')  # noqa: E501
avg_diff_standard_deviation = AggregateColumnComparisonAdapter(DiffColumnMetricAdapter(StandardDeviation()), summary_type='avg')  # noqa: E501
diff_min_kendell_tau_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(KendellTauCorrelation(), summary_type='min'))  # noqa: E501
diff_max_kendell_tau_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(KendellTauCorrelation(), summary_type='max'))  # noqa: E501
diff_avg_kendell_tau_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(KendellTauCorrelation(), summary_type='avg'))  # noqa: E501
min_diff_kendell_tau_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(KendellTauCorrelation()), summary_type='min')  # noqa: E501
max_diff_kendell_tau_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(KendellTauCorrelation()), summary_type='max')  # noqa: E501
avg_diff_kendell_tau_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(KendellTauCorrelation()), summary_type='avg')  # noqa: E501
diff_min_spearman_rho_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(SpearmanRhoCorrelation(), summary_type='min'))  # noqa: E501
diff_max_spearman_rho_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(SpearmanRhoCorrelation(), summary_type='max'))  # noqa: E501
diff_avg_spearman_rho_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(SpearmanRhoCorrelation(), summary_type='avg'))  # noqa: E501
min_diff_spearman_rho_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(SpearmanRhoCorrelation()), summary_type='min')  # noqa: E501
max_diff_spearman_rho_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(SpearmanRhoCorrelation()), summary_type='max')  # noqa: E501
avg_diff_spearman_rho_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(SpearmanRhoCorrelation()), summary_type='avg')  # noqa: E501
diff_min_cramers_v = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(CramersV(), summary_type='min'))
diff_max_cramers_v = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(CramersV(), summary_type='max'))
diff_avg_cramers_v = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(CramersV(), summary_type='avg'))
min_diff_cramers_v = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(CramersV()), summary_type='min')
max_diff_cramers_v = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(CramersV()), summary_type='max')
avg_diff_cramers_v = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(CramersV()), summary_type='avg')
min_kolmogorov_smirnov_distance = AggregateColumnComparisonAdapter(KolmogorovSmirnovDistance(), summary_type='min')
max_kolmogorov_smirnov_distance = AggregateColumnComparisonAdapter(KolmogorovSmirnovDistance(), summary_type='max')
avg_kolmogorov_smirnov_distance = AggregateColumnComparisonAdapter(KolmogorovSmirnovDistance(), summary_type='avg')
min_earth_movers_distance = AggregateColumnComparisonAdapter(EarthMoversDistance(), summary_type='min')
max_earth_movers_distance = AggregateColumnComparisonAdapter(EarthMoversDistance(), summary_type='max')
avg_earth_movers_distance = AggregateColumnComparisonAdapter(EarthMoversDistance(), summary_type='avg')
diff_min_categorical_logistic_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(CategoricalLogisticR2(), summary_type='min'))  # noqa: E501
diff_max_categorical_logistic_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(CategoricalLogisticR2(), summary_type='max'))  # noqa: E501
diff_avg_categorical_logistic_correlation = DiffDataFrameMetricAdapter(AggregateTwoColumnMetricAdapter(CategoricalLogisticR2(), summary_type='avg'))  # noqa: E501
min_diff_categorical_logistic_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(CategoricalLogisticR2()), summary_type='min')  # noqa: E501
max_diff_categorical_logistic_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(CategoricalLogisticR2()), summary_type='max')  # noqa: E501
avg_diff_categorical_logistic_correlation = AggregateTwoColumnComparisonAdapter(DiffTwoColumnMetricAdapter(CategoricalLogisticR2()), summary_type='avg')  # noqa: E501
diff_predictive_modelling_score = DiffDataFrameMetricAdapter(PredictiveModellingScore())
predictive_modelling_comparison = PredictiveModellingComparison()

# DataFrameMetricVectors
# -----------------------------------------------------------------------------
diff_vector = DiffVector()
fractional_diff_vector = FractionalDiffVector()

# DataFrameMetricVectors
# -----------------------------------------------------------------------------
standard_deviation_vector = ColumnMetricVector(StandardDeviation())

# DataFrameComparisonVectors
# -----------------------------------------------------------------------------
kolmogorov_smirnov_distance_vector = ColumnComparisonVector(KolmogorovSmirnovDistance())
earth_movers_distance_vector = ColumnComparisonVector(EarthMoversDistance())

# DataFrameMetricMatrices
# -----------------------------------------------------------------------------
kendell_tau_correlation_matrix = TwoColumnMetricMatrix(KendellTauCorrelation())
spearman_rho_correlation_matrix = TwoColumnMetricMatrix(SpearmanRhoCorrelation())
cramers_v_matrix = TwoColumnMetricMatrix(CramersV())
categorical_logistic_correlation_matrix = TwoColumnMetricMatrix(CategoricalLogisticR2())

# DataFrameComparisonMatrices
# -----------------------------------------------------------------------------
diff_kendell_tau_correlation_matrix = TwoColumnComparisonMatrix(DiffTwoColumnMetricAdapter(KendellTauCorrelation()))
diff_spearman_rho_correlation_matrix = TwoColumnComparisonMatrix(DiffTwoColumnMetricAdapter(SpearmanRhoCorrelation()))
diff_cramers_v_matrix = TwoColumnComparisonMatrix(DiffTwoColumnMetricAdapter(CramersV()))
diff_categorical_logistic_correlation_matrix = TwoColumnComparisonMatrix(DiffTwoColumnMetricAdapter(CategoricalLogisticR2()))  # noqa: E501

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
    'ALL', 'COLUMN_METRICS', 'TWO_COLUMN_METRICS', 'DATA_FRAME_METRICS', 'COLUMN_COMPARISONS',
    'TWO_COLUMN_COMPARISONS', 'DATA_FRAME_COMPARISONS', 'ColumnMetric', 'TwoColumnMetric', 'DataFrameMetric',
    'ColumnComparison', 'TwoColumnComparison', 'DataFrameComparison', 'DataFrameVector',
    'DataFrameComparisonVector', 'DataFrameMatrix', 'DataFrameComparisonMatrix', 'RollingColumnMetricVector',
    'ModellingMetric', 'ClassificationMetric', 'RegressionMetric', 'ClassificationPlotMetric'
]
__all__.extend([metric_name for metric_name in _Metric.ALL.keys()])
