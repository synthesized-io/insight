from .metrics import (CategoricalLogisticR2, CramersV, EarthMoversDistance, KendallTauCorrelation,
                      KolmogorovSmirnovDistance, Mean, SpearmanRhoCorrelation, StandardDeviation)
from .metrics_base import (ChainColumnVector, ClassificationMetric, ClassificationPlotMetric, ColumnComparisonVector,
                           ColumnMetric, ColumnMetricVector, ColumnVector, DataFrameMatrix, DataFrameMetric,
                           DataFrameVector, DiffColumnMetricAdapter, DiffMetricMatrix, ModellingMetric,
                           RegressionMetric, RollingColumnMetricVector, TwoColumnMetric, TwoColumnMetricMatrix,
                           TwoDataFrameMatrix, TwoDataFrameMetric, TwoDataFrameVector)
from .modelling_metrics import (ROC_AUC, Accuracy, ConfusionMatrix, F1Score, MeanAbsoluteError, MeanSquaredError,
                                PR_Curve, Precision, R2_Score, Recall, ROC_Curve, predictive_modelling_comparison,
                                predictive_modelling_score)
from .vectors import DiffVector, FractionalDiffVector


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
    'ColumnMetric', 'ColumnMetricVector', 'ColumnComparisonVector',
    'TwoColumnMetric', 'DataFrameMetric', 'TwoDataFrameMetric', 'DataFrameVector',
    'TwoDataFrameVector', 'DataFrameMatrix', 'TwoDataFrameMatrix', 'DiffMetricMatrix', 'RollingColumnMetricVector',
    'DiffVector', 'FractionalDiffVector', 'ChainColumnVector', 'ColumnVector', 'TwoColumnMetricMatrix',
    'ModellingMetric', 'ClassificationMetric', 'RegressionMetric', 'ClassificationPlotMetric',
    'DiffColumnMetricAdapter', 'predictive_modelling_score', 'predictive_modelling_comparison',
    'Mean', 'StandardDeviation', 'KendallTauCorrelation', 'SpearmanRhoCorrelation', 'CramersV',
    'KolmogorovSmirnovDistance', 'EarthMoversDistance', 'CategoricalLogisticR2'
]
