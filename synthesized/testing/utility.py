# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# The testing environment
"""This module contains tools for utility testing."""

from __future__ import division, print_function, absolute_import

from enum import Enum
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.scorer import roc_auc_scorer
from statsmodels.formula.api import ols, mnlogit

from synthesized.highdim import HighDimSynthesizer
from synthesized.common.values import CategoricalValue
from synthesized.common.values import ContinuousValue
from synthesized.common.values import DateValue
from synthesized.common.values import SamplingValue
from synthesized.common.values import Value

COLOR_ORIG = '#00AB26'
COLOR_SYNTH = '#2794F3'


class DisplayType(Enum):
    """Used to display columns differently based on their type."""

    CATEGORICAL = 1
    CATEGORICAL_SIMILARITY = 2
    CONTINUOUS = 3


class UtilityTesting:
    """A universal set of utilities that let you to compare quality of original vs synthetic data."""

    def __init__(self,
                 synthesizer: HighDimSynthesizer,
                 df_orig: pd.DataFrame,
                 df_test: pd.DataFrame,
                 df_synth: pd.DataFrame):
        """Create an instance of UtilityTesting.

        Args:
            synthesizer: A synthesizer instance.
            df_orig: A DataFrame with original data that was used for training
            df_test: A DataFrame with hold-out original data
            df_synth: A DataFrame with synthetic data
        """
        self.df_orig = df_orig.copy()
        self.df_test = df_test.copy()
        self.df_synth = df_synth.copy()

        self.df_orig_encoded = synthesizer.preprocess(df=df_orig)
        self.df_test_encoded = synthesizer.preprocess(df=df_test)
        self.df_synth_encoded = synthesizer.preprocess(df=df_synth)

        self.display_types: Dict[str, DisplayType] = {}
        for value in synthesizer.values:
            if isinstance(value, DateValue):
                self.df_orig[value.name] = pd.to_datetime(self.df_orig[value.name])
                self.df_test[value.name] = pd.to_datetime(self.df_test[value.name])
                self.df_synth[value.name] = pd.to_datetime(self.df_synth[value.name])
            elif isinstance(value, ContinuousValue):
                self.display_types[value.name] = DisplayType.CONTINUOUS
            elif isinstance(value, CategoricalValue):
                self.display_types[value.name] = DisplayType.CATEGORICAL
        self.value_by_name: Dict[str, Value] = {}
        for v in synthesizer.values:
            self.value_by_name[v.name] = v

    def show_corr_matrices(self, figsize: Tuple[float, float] = (15, 11)) -> None:
        """Plot two correlations matrices: one for the original data and one for the synthetic one.

        Args:
            figsize: width, height in inches.
        """
        def show_corr_matrix(df, title=None, ax=None):
            sns.set(style='white')

            # Compute the correlation matrix
            corr = df.corr()

            # Generate a mask for the upper triangle
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            hm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                             square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax)

            if title is not None:
                hm.set_title(title)

        # Set up the matplotlib figure
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        show_corr_matrix(self.df_test, title='Original', ax=ax1)
        show_corr_matrix(self.df_synth, title='Synthetic', ax=ax2)

    def show_corr_distances(self, figsize: Tuple[float, float] = (4, 10)) -> None:
        """Plot a barplot with correlation diffs between original anf synthetic columns.

        Args:
            figsize: width, height in inches.
        """
        distances = (self.df_test.corr() - self.df_synth.corr()).abs()
        result = []
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                if i < j:
                    row_name = distances.index[i]
                    col_name = distances.iloc[:, j].name
                    result.append({'column': '{} / {}'.format(row_name, col_name), 'distance': distances.iloc[i, j]})
        if not result:
            return
        df = pd.DataFrame.from_records(result)
        print('Average distance:', df['distance'].mean())
        print('Max distance:', df['distance'].max())
        plt.figure(figsize=figsize)
        g = sns.barplot(y='column', x='distance', data=df)
        g.set_xlim(0.0, 1.0)

    @staticmethod
    def _filter_column_data_types(data_frame):
        categorical, continuous = [], []
        for name, dtype in data_frame.dtypes.items():
            if dtype.kind == 'f':
                continuous.append(name)
            elif dtype.kind == 'O':
                categorical.append(name)
        return categorical, continuous

    def show_anova(self, figsize: Tuple[float, float] = (10, 10)):
        """
        Plot a heatmap with ANOVA R^2 between all pairs of categorical columns
        for original and another synthetic datasets.

        The R^2 is to be interpreted as a measure of association between two columns.

        Args:
            figsize: width, height in inches.
        """
        categorical, continuous = self._filter_column_data_types(data_frame=self.df_synth)
        df = pd.concat([self.df_test.assign(source='orig'), self.df_synth.assign(source='synth')]).reset_index()
        orig_anovas = np.zeros((len(categorical), len(continuous)))
        synth_anovas = np.zeros((len(categorical), len(continuous)))
        for i_cat, cat_name in enumerate(categorical):
            for i_cont, cont_name in enumerate(continuous):
                df_orig = df[df['source'] == 'orig']
                orig = ols(f"{cont_name} ~ C({cat_name})", data=df_orig).fit()
                orig_anovas[i_cat,  i_cont] = orig.rsquared

                df_synth = df[df['source'] == 'synth']
                synth = ols(f"{cont_name} ~ C({cat_name})", data=df_synth).fit()
                synth_anovas[i_cat, i_cont] = synth.rsquared
        orig_anovas = pd.DataFrame(orig_anovas, index=categorical,  columns=continuous)
        synth_anovas = pd.DataFrame(synth_anovas, index=categorical,  columns=continuous)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        _ = sns.heatmap(orig_anovas, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                        square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax1)

        _ = sns.heatmap(synth_anovas, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                        square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax2)

    def show_categorical_rsquared(self, figsize: Tuple[float, float] = (10, 10)):
        """
        Plot a heatmap with multinomial regression R^2 between all pairs of categorical columns
        for original and another synthetic datasets.

        The R^2 is to be interpreted as a measure of association between two columns.

        Args:
            figsize: width, height in inches.
        """
        categorical, _ = self._filter_column_data_types(data_frame=self.df_synth)
        source = pd.DataFrame({'source': len(self.df_test)*['orig'] + len(self.df_synth)*['synth']}).reset_index()
        orig_synth = pd.concat([self.df_test, self.df_synth]).reset_index(inplace=False)
        df = pd.concat([orig_synth, source], axis=1)
        orig_anovas = np.zeros((len(categorical), len(categorical)))
        synth_anovas = np.zeros((len(categorical), len(categorical)))
        for i_cat in range(len(categorical)):
            i_cat_name = categorical[i_cat]
            orig_anovas[i_cat, i_cat] = 1.0
            synth_anovas[i_cat, i_cat] = 1.0
            for j_cat in range(i_cat):
                j_cat_name = categorical[j_cat]
                orig = mnlogit(f"{i_cat_name} ~ C({j_cat_name})", data=df[df['source'] == 'orig']).fit(method='cg')
                orig_anovas[i_cat, j_cat] = orig.prsquared

                synth = mnlogit(f"{i_cat_name} ~ C({j_cat_name})", data=df[df['source'] == 'synth']).fit(method='cg')
                synth_anovas[i_cat, j_cat] = synth.prsquared

        orig_anovas = pd.DataFrame(orig_anovas, index=categorical, columns=categorical)
        synth_anovas = pd.DataFrame(synth_anovas, index=categorical, columns=categorical)

        mask = np.zeros_like(orig_anovas, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        _ = sns.heatmap(orig_anovas, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                        square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax1)

        _ = sns.heatmap(synth_anovas, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                        square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax2)

    def show_distributions(self,
                           remove_outliers: float = 0.0,
                           figsize: Tuple[float, float] = (14, 50),
                           cols: int = 2) -> None:
        """Plot comparison plots of all variables in the original and synthetic datasets.

        Args:
            remove_outliers: Percent of outliers to remove.
            figsize: width, height in inches.
            cols: Number of columns in the plot grid.
        """
        concatenated = pd.concat([self.df_test.assign(dataset='orig'), self.df_synth.assign(dataset='synth')])
        fig = plt.figure(figsize=figsize)
        for i, (col, dtype) in enumerate(self.display_types.items()):
            ax = fig.add_subplot(len(self.display_types), cols, i + 1)
            if dtype == DisplayType.CATEGORICAL:
                ax = sns.countplot(x=col, hue='dataset', data=concatenated,
                                   palette={'orig': COLOR_ORIG, 'synth': COLOR_SYNTH}, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            elif dtype == DisplayType.CONTINUOUS:
                percentiles = [remove_outliers * 100. / 2, 100 - remove_outliers * 100. / 2]
                start, end = np.percentile(self.df_test[col], percentiles)
                # workaround for kde failing on datasets with only one value
                if self.df_test[col].nunique() < 2 or self.df_synth[col].nunique() < 2:
                    kde = False
                else:
                    kde = True
                sns.distplot(self.df_test[col], color=COLOR_ORIG, label='orig', kde=kde, kde_kws={'clip': (start, end)},
                             hist_kws={'color': COLOR_ORIG, 'range': [start, end]}, ax=ax)
                sns.distplot(self.df_synth[col], color=COLOR_SYNTH, label='synth', kde=kde,
                             kde_kws={'clip': (start, end)},
                             hist_kws={'color': COLOR_SYNTH, 'range': [start, end]}, ax=ax)
            plt.legend()

    def utility(self,
                target: str,
                classifier: BaseEstimator = GradientBoostingClassifier(),
                regressor: BaseEstimator = GradientBoostingRegressor()) -> float:
        """Compute utility.

        Utility is a score of estimator trained on synthetic data.

        Args:
            target: Response variable
            classifier: If target a categorical then this estimator will be used.
            regressor: If target a numerical then this estimator will be used.

        Returns: Utility score.
        """
        def skip_sampling(df):
            for col in df.columns:
                if col not in self.value_by_name or isinstance(self.value_by_name[col], SamplingValue):
                    df = df.drop(col, axis=1)
            return df

        X, y = self.df_orig_encoded.drop(target, 1), self.df_orig_encoded[target]
        X_synth, y_synth = self.df_synth_encoded.drop(target, 1), self.df_synth_encoded[target]
        X_test, y_test = self.df_test_encoded.drop(target, 1), self.df_test_encoded[target]

        X = skip_sampling(X)
        X_synth = skip_sampling(X_synth)
        X_test = skip_sampling(X_test)

        if self.display_types[target] == DisplayType.CATEGORICAL:
            clf = clone(classifier)
            clf.fit(X, y)
            orig_score = roc_auc_scorer(clf, X_test, y_test)

            clf = clone(classifier)
            clf.fit(X_synth, y_synth)
            synth_score = roc_auc_scorer(clf, X_test, y_test)

            print("ROC AUC (orig):", orig_score)
            print("ROC AUC (synth):", synth_score)
            return synth_score
        else:
            clf = clone(regressor)
            clf.fit(X, y)
            y_pred_orig = clf.predict(X_test)

            clf = clone(regressor)
            clf.fit(X_synth, y_synth)
            y_pred_synth = clf.predict(X_test)

            orig_score = r2_score(y_test, y_pred_orig)
            synth_score = r2_score(y_test, y_pred_synth)
            print("R2 (orig):", orig_score)
            print("R2 (synth):", synth_score)
            return synth_score

    def autocorrelation_diff_plot_seaborn(self, max_lag: int = 100) -> None:
        """Plot autocorrelation.

        Args:
            max_lag: A max lag
        """
        # for synthetic data at the moment, TODO for real data
        # how do we detect time column?
        def autocorrelation(h, data, mean, n, c0):
            return ((data[:n - h] - mean) *
                    (data[h:] - mean)).sum() / float(n) / c0

        n_orig, n_synth = len(self.df_test), len(self.df_synth)
        original_data, synthetic_data = np.asarray(self.df_test), np.asarray(self.df_synth)

        mean_orig, mean_synth = np.mean(original_data), np.mean(synthetic_data)
        c0_orig = np.sum((original_data - mean_orig) ** 2) / float(n_orig)
        c0_synth = np.sum((synthetic_data - mean_synth) ** 2) / float(n_synth)

        n = min(n_orig, n_synth, max_lag)
        x = np.arange(n) + 1

        y_orig = [autocorrelation(loc, original_data[:n], mean_orig, n, c0_orig) for loc in x]
        y_synth = [autocorrelation(loc, synthetic_data, mean_synth, n_synth, c0_synth) for loc in x]

        sns.set(style='whitegrid')

        data = pd.DataFrame({'Original': y_orig, 'Synthetic': y_synth})
        sns.lineplot(data=data, palette=[COLOR_SYNTH, COLOR_ORIG], linewidth=2.5)
        return mean_squared_error(y_orig, y_synth)

    def show_distribution_distances(self):
        """Plot a barplot with KS-distances between original and synthetic columns."""
        result = []
        for col in self.df_test.columns.values:
            distance = ks_2samp(self.df_test[col], self.df_synth[col])[0]
            result.append({'column': col, 'distance': distance})
        df = pd.DataFrame.from_records(result)
        print("Average distance:", df['distance'].mean())
        print("Max distance:", df['distance'].max())
        g = sns.barplot(y='column', x='distance', data=df)
        g.set_xlim(0.0, 1.0)
