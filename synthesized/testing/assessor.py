import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import ClassifierMixin

from synthesized.insight.metrics.metrics import (CategoricalLogisticR2, CramersV, EarthMoversDistance,
                                                 KendallTauCorrelation, KolmogorovSmirnovDistance,
                                                 SpearmanRhoCorrelation)

from .plotting import (plot_classification_metrics, plot_classification_metrics_test, plot_standard_metrics,
                       set_plotting_style, show_distributions, show_first_order_metric_distances,
                       show_second_order_metric_distances, show_second_order_metric_matrices)
from ..insight.metrics import TwoColumnMetric
from ..insight.metrics.modelling_metrics import predictive_modelling_comparison
from ..metadata import DataFrameMeta
from ..model.factory import ModelFactory

logger = logging.getLogger(__name__)


class Assessor:
    """A universal set of utilities that let you to assess the quality of synthetic against original data."""

    def __init__(self, df_meta: DataFrameMeta):
        """Create an instance of Assessor.

        Args:
            synthesizer: A synthesizer instance.
            df_orig: A DataFrame with original data that was used for training
            df_test: A DataFrame with hold-out original data
            df_synth: A DataFrame with synthetic data
        """
        self.df_meta = df_meta
        self.df_model = ModelFactory()(df_meta)

        # Set the style of plots
        set_plotting_style()

    def show_standard_metrics(self, df_test: pd.DataFrame, df_synth: pd.DataFrame, ax: plt.Axes = None):
        """Plot average and maximum distances for standard metrics.

        Standard metrics are:
            * Kolmogorov–Smirnov distance.
            * Earth Mover's distance.
            * Kendall's Tau correlation distance.
            * Cramer's V correlation distance.
            * Logistic R^2 score distance.

        Args:
            df_test: Original test dataset.
            df_synth: Synthesized dataset.
            ax: Axes to plot data on.
        """
        current_result = plot_standard_metrics(df_test, df_synth, self.df_meta, ax=ax)
        return current_result

    def show_distributions(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                           remove_outliers: float = 0.01, figsize: Tuple[float, float] = None,
                           cols: int = 2, sample_size: int = 10_000) -> None:
        """Plot comparison plots of marginal distributions of all columns in the original and synthetic datasets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
            remove_outliers: Percent of outliers to remove for better visualization purposes.
            figsize: width, height in inches.
            cols: Number of columns in the plot grid.
            sample_size: Maximum sample size to show distributions.
        """
        show_distributions(df_orig, df_synth, self.df_model, remove_outliers=remove_outliers, figsize=figsize,
                           cols=cols, sample_size=sample_size)

    def show_first_order_metric_distances(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                                          metric: TwoColumnMetric) -> Tuple[float, float]:
        """Plot and compare distribution distances for each column between the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
            metric: the distribution distance metric to show.

        Returns the maximum and average distances.
        """

        dist_max, dist_avg = show_first_order_metric_distances(df_orig=df_orig, df_synth=df_synth,
                                                               df_model=self.df_model, metric=metric)

        logger.info(f"Maximum {metric.name} dist:\t{dist_max:.2f}")
        logger.info(f"Average {metric.name} dist:\t{dist_avg:.2f}")
        return dist_max, dist_avg

    def show_ks_distances(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame):
        """Plot and compare Kolmogorov-Smirnov distances for each column between the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.

        Returns the maximum and average distances.
        """
        return self.show_first_order_metric_distances(df_orig, df_synth, KolmogorovSmirnovDistance())

    def show_emd_distances(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame):
        """Plot and compare Earth Mover's distances for each column between the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.

        Returns the maximum and average distances.
        """
        return self.show_first_order_metric_distances(df_orig, df_synth, EarthMoversDistance())

    def show_second_order_metric_matrices(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame, metric: TwoColumnMetric):
        """Plot two correlations matrices, one for the original data and one for the synthetic one.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
            metric: the correlation metric to show.
        """
        show_second_order_metric_matrices(df_orig, df_synth, df_model=self.df_model, metric=metric)

    def show_kendall_tau_matrices(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame, max_p_value: float = 1.0):
        """Plot Kendall's Tau correlation matrices for both the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
            max_p_value: If the p-value is higher than this for any pair of columns, will show an empty cell.
        """
        self.show_second_order_metric_matrices(
            df_orig=df_orig, df_synth=df_synth, metric=KendallTauCorrelation(max_p_value=max_p_value)
        )

    def show_spearman_rho_matrices(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame, max_p_value: float = 1.0):
        """Plot Spearman's Rho correlation matrices for both the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
            max_p_value: If the p-value is higher than this for any pair of columns, will show an empty cell.
        """
        self.show_second_order_metric_matrices(
            df_orig=df_orig, df_synth=df_synth, metric=SpearmanRhoCorrelation(max_p_value=max_p_value)
        )

    def show_cramers_v_matrices(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame):
        """Plot Cramer's V matrices for both the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
        """
        self.show_second_order_metric_matrices(df_orig=df_orig, df_synth=df_synth, metric=CramersV())

    def show_categorical_logistic_r2_matrices(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame):
        """Plot R^2 score matrices for both the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
        """
        self.show_second_order_metric_matrices(df_orig=df_orig, df_synth=df_synth, metric=CategoricalLogisticR2())

    def show_second_order_metric_distances(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                                           metric: TwoColumnMetric) -> Tuple[float, float]:
        """Plot and compare the given correlation metric distances between the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
            metric: Correlation metric to show distances on.

        Returns the maximum and average correlation distances.
        """
        corr_dist_max, corr_dist_avg = show_second_order_metric_distances(df_orig, df_synth,
                                                                          df_model=self.df_model, metric=metric)

        logger.info(f"Maximum {metric.name} dist:\t{corr_dist_max:.2f}")
        logger.info(f"Average {metric.name} dist:\t{corr_dist_avg:.2f}")

        return corr_dist_max, corr_dist_avg

    def show_kendall_tau_distances(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame, max_p_value: float = 1.0):
        """Plot and compare Kendall's Tau correlation distances between the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.

        Returns the maximum and average correlation distances.
        """
        return self.show_second_order_metric_distances(
            df_orig=df_orig, df_synth=df_synth, metric=KendallTauCorrelation(max_p_value=max_p_value)
        )

    def show_spearman_rho_distances(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame, max_p_value: float = 1.0):
        """Plot and compare Spearman's Rho correlation distances between the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.

        Returns the maximum and average correlation distances.
        """
        return self.show_second_order_metric_distances(
            df_orig=df_orig, df_synth=df_synth, metric=SpearmanRhoCorrelation(max_p_value=max_p_value)
        )

    def show_cramers_v_distances(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame):
        """Plot and compare Cramer's V correlation distances between the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.

        Returns the maximum and average correlation distances.
        """
        return self.show_second_order_metric_distances(df_orig=df_orig, df_synth=df_synth, metric=CramersV())

    def show_categorical_logistic_r2_distances(self, df_orig: pd.DataFrame,
                                               df_synth: pd.DataFrame) -> Tuple[float, float]:
        """Plot and compare R^2 score distances between the original and synthesized sets.

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.

        Returns the maximum and average correlation distances.
        """
        return self.show_second_order_metric_distances(df_orig=df_orig, df_synth=df_synth, metric=CategoricalLogisticR2())

    def plot_classification_metrics(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame, target: str,
                                    df_test: pd.DataFrame, clf: ClassifierMixin,
                                    names: Optional[Tuple[str, str]] = None):
        """Plot ROC curve, PR curve and Confusion Matrix for the given classifier trained on two dataframes
        (``df_orig`` and ``df_synth``), and evaluated on the same dataset (``df_test``).

        Args:
            df_orig: Original dataset, or first training set.
            df_synth: Synthesized dataset, or second training set.
            target: Response variable.
            df_test: Test dataset.
            clf: The classifier to be used.
            names: Names used in plots to identify ``df_orig`` and ``df_synth``, ('Train Set 1', 'Train Set 2') by default.
        """

        plot_classification_metrics(df_model=self.df_model, target=target, df_train1=df_orig, df_train2=df_synth,
                                    df_test=df_test, clf=clf, names=names)

    def plot_classification_metrics_test(self, df_train: pd.DataFrame, df_test1: pd.DataFrame, df_test2: pd.DataFrame,
                                         target: str, clf: ClassifierMixin, names: Optional[Tuple[str, str]] = None):
        """Plot ROC curve, PR curve and Confusion Matrix for the given classifier trained on the same dataframe
        (``df_train``), and evaluated on two datasets (``df_test1`` and ``df_test2``).

        Args:
            df_train: Training set.
            df_test1: First test dataset.
            df_test2: Second test dataset.
            target: Response variable.
            clf: The classifier to be used.
            names: Names used in plots to identify ``df_test1`` and ``df_test2``, ('Test Set 1', 'Test Set 2') by default.
        """

        plot_classification_metrics_test(df_model=self.df_model, target=target, df_train=df_train, df_test1=df_test1,
                                         df_test2=df_test2, clf=clf, names=names)

    def utility(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                target: str, model: str = 'GradientBoosting') -> float:
        """Compute utility, a score of estimator trained on synthetic data.

        The score to be computed will depend on the type of target distribution, ROC AUC for categorical and R2 for
        continuous. Estimator can be one of the following:
        * 'Linear'
        * 'Logistic'
        * 'LinearSVM'
        * 'GradientBoosting'
        * 'RandomForest'
        * 'MLP'

        Args:
            df_orig: Original dataset.
            df_synth: Synthesized dataset.
            target: Response variable.
            model: The estimator to use (converted to classifier or regressor).

        Returns: Utility score.
        """
        x_labels = list(filter(lambda v: v != target, df_orig.columns))
        orig_score, synth_score, metric, _ = predictive_modelling_comparison(
            df_orig, df_synth, model=model, y_label=target, x_labels=x_labels
        )

        logger.info(f"{metric} (orig): {orig_score:.2f}")
        logger.info(f"{metric} (synth): {synth_score:.2f}")

        return synth_score
