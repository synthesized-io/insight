# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# The testing environment
"""This module contains tools for utility testing."""

from __future__ import division, print_function, absolute_import

from enum import Enum
from typing import Tuple, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, normalized_mutual_info_score
from sklearn.metrics import roc_auc_score
from statsmodels.formula.api import ols, mnlogit
from statsmodels.tsa.stattools import acf, pacf

from ..common.synthesizer import Synthesizer
from ..common.values import CategoricalValue
from ..common.values import ContinuousValue
from ..common.values import DateValue
from ..common.values import DecomposedContinuousValue
from ..common.values import NanValue
from ..common.values import SamplingValue
from ..common.values import Value
from ..testing import metrics as eval_metrics
from ..testing.util import categorical_emd

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
                 synthesizer: Synthesizer,
                 df_orig: pd.DataFrame,
                 df_test: pd.DataFrame,
                 df_synth: pd.DataFrame,
                 identifier=None):
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

        self.date_cols: List[str] = []
        self.continuous_cols: List[str] = []
        self.categorical_cols: List[str] = []

        self.display_types: Dict[str, DisplayType] = {}
        for value in synthesizer.get_values():
            if isinstance(value, NanValue):
                value = value.value

            if isinstance(value, DateValue):
                self.df_orig[value.name] = pd.to_datetime(self.df_orig[value.name])
                self.df_test[value.name] = pd.to_datetime(self.df_test[value.name])
                self.df_synth[value.name] = pd.to_datetime(self.df_synth[value.name])
                self.date_cols.append(value.name)
            elif isinstance(value, ContinuousValue) or isinstance(value, DecomposedContinuousValue):
                self.display_types[value.name] = DisplayType.CONTINUOUS
                self.continuous_cols.append(value.name)
            elif isinstance(value, CategoricalValue):
                self.display_types[value.name] = DisplayType.CATEGORICAL
                self.categorical_cols.append(value.name)
        self.value_by_name: Dict[str, Value] = {}
        for v in synthesizer.get_values():
            self.value_by_name[v.name] = v

        # Identifiers (only for Time-Series)
        self.identifier = identifier
        if identifier:
            self.unique_ids_orig = self.df_orig[identifier].unique()
            self.unique_ids_synth = self.df_synth[identifier].unique()
        else:
            self.unique_ids_orig = []
            self.unique_ids_synth = []

        # Set the style of plots
        plt.style.use('seaborn')
        mpl.rcParams["axes.facecolor"] = 'w'
        mpl.rcParams['grid.color'] = 'grey'
        mpl.rcParams['grid.alpha'] = 0.1

        mpl.rcParams['axes.linewidth'] = 0.3
        mpl.rcParams['axes.edgecolor'] = 'grey'

        mpl.rcParams['axes.spines.right'] = True
        mpl.rcParams['axes.spines.top'] = True

    def show_corr_matrices(self, figsize: Tuple[float, float] = (15, 11)) -> None:
        """Plot two correlations matrices: one for the original data and one for the synthetic one.

        Args:
            figsize: width, height in inches.
        """

        def show_corr_matrix(df, title=None, ax=None):
            sns.set(style='white')

            df_numeric = df.copy()
            for c in df_numeric.columns:
                df_numeric[c] = pd.to_numeric(df_numeric[c], errors='coerce')
                if df_numeric[c].isna().all():
                    df_numeric.drop(c, axis=1, inplace=True)

            # Compute the correlation matrix
            corr = df_numeric.corr(method='kendall')

            # Generate a mask for the upper triangle
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            hm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                             square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax, annot=True, fmt='.2f')

            if ax:
                ax.set_ylim(ax.get_ylim()[0] + .5, ax.get_ylim()[1] - .5)
            if title:
                hm.set_title(title)

        # Set up the matplotlib figure
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        plt.title('Correlation Matrices')

        if self.identifier:
            show_corr_matrix(self.df_test.drop(self.identifier, axis=1), title='Original', ax=ax1)
            show_corr_matrix(self.df_synth.drop(self.identifier, axis=1), title='Synthetic', ax=ax2)
        else:
            show_corr_matrix(self.df_test, title='Original', ax=ax1)
            show_corr_matrix(self.df_synth, title='Synthetic', ax=ax2)
        plt.show()

    def show_corr_distances(self, figsize: Tuple[float, float] = None) -> Tuple[float, float]:
        """Plot a barplot with correlation diffs between original anf synthetic columns.

        Args:
            figsize: width, height in inches.
        """
        if self.identifier:
            distances = (self.df_test.drop(self.identifier, axis=1).corr(method='kendall') -
                         self.df_synth.drop(self.identifier, axis=1).corr(method='kendall')).abs()
        else:
            distances = (self.df_test.corr(method='kendall') - self.df_synth.corr(method='kendall')).abs()

        result = []
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                if i < j:
                    row_name = distances.index[i]
                    col_name = distances.iloc[:, j].name
                    result.append({'column': '{} / {}'.format(row_name, col_name), 'distance': distances.iloc[i, j]})

        if not result:
            return 0., 0.

        df = pd.DataFrame.from_records(result)
        if figsize is None:
            figsize = (10, len(df) // 6 + 2)

        corr_dist_max = df['distance'].max()
        corr_dist_avg = df['distance'].mean()

        print('Max correlation distance:', corr_dist_max)
        print('Average correlation distance:', corr_dist_avg)

        plt.figure(figsize=figsize)
        plt.title('Correlation Distances')
        g = sns.barplot(y='column', x='distance', data=df)
        g.set_xlim(0.0, 1.0)
        plt.show()

        return corr_dist_max, corr_dist_avg

    def _filter_column_data_types(self):
        categorical, continuous = [], []
        for name, dtype in self.display_types.items():
            if dtype == DisplayType.CONTINUOUS:
                continuous.append(name)
            elif dtype == DisplayType.CATEGORICAL:
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
        categorical, continuous = self._filter_column_data_types()
        df = pd.concat([self.df_test.assign(source='orig'), self.df_synth.assign(source='synth')]).reset_index()
        orig_anovas = np.zeros((len(categorical), len(continuous)))
        synth_anovas = np.zeros((len(categorical), len(continuous)))
        for i_cat, cat_name in enumerate(categorical):
            for i_cont, cont_name in enumerate(continuous):
                df_orig = df[df['source'] == 'orig']
                orig = ols(f"{cont_name} ~ C({cat_name})", data=df_orig).fit()
                orig_anovas[i_cat, i_cont] = orig.rsquared

                df_synth = df[df['source'] == 'synth']
                synth = ols(f"{cont_name} ~ C({cat_name})", data=df_synth).fit()
                synth_anovas[i_cat, i_cont] = synth.rsquared
        orig_anovas = pd.DataFrame(orig_anovas, index=categorical, columns=continuous)
        synth_anovas = pd.DataFrame(synth_anovas, index=categorical, columns=continuous)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        _ = sns.heatmap(orig_anovas, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                        square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax1)

        _ = sns.heatmap(synth_anovas, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                        square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax2)
        plt.show()

    def show_categorical_rsquared(self, figsize: Tuple[float, float] = (10, 10)):
        """
        Plot a heatmap with multinomial regression R^2 between all pairs of categorical columns
        for original and another synthetic datasets.

        The R^2 is to be interpreted as a measure of association between two columns.

        Args:
            figsize: width, height in inches.
        """
        categorical, _ = self._filter_column_data_types()
        source = pd.DataFrame({'source': len(self.df_test) * ['orig'] + len(self.df_synth) * ['synth']}).reset_index()
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

        plt.show()

    def show_distributions(self,
                           remove_outliers: float = 0.0,
                           figsize: Tuple[float, float] = None,
                           cols: int = 2, sample_size: int = 10_000) -> None:
        """Plot comparison plots of all variables in the original and synthetic datasets.

        Args:
            remove_outliers: Percent of outliers to remove.
            figsize: width, height in inches.
            cols: Number of columns in the plot grid.
        """
        if not figsize:
            figsize = (14, 5 * len(self.display_types) + 2)

        fig = plt.figure(figsize=figsize)
        for i, (col, dtype) in enumerate(self.display_types.items()):
            ax = fig.add_subplot(len(self.display_types), cols, i + 1)
            title = col

            col_test = self.df_orig[col].dropna()
            col_synth = self.df_synth[col].dropna()
            if len(col_test) == 0 or len(col_synth) == 0:
                continue

            if dtype == DisplayType.CATEGORICAL:

                df_col_test = pd.DataFrame(col_test)
                df_col_synth = pd.DataFrame(col_synth)

                # We sample orig and synth them so that they have the same size to make the plots more comprehensive
                sample_size = min(sample_size, len(col_test), len(col_synth))
                concatenated = pd.concat([df_col_test.assign(dataset='orig').sample(sample_size),
                                          df_col_synth.assign(dataset='synth').sample(sample_size)])

                ax = sns.countplot(x=col, hue='dataset', data=concatenated,
                                   palette={'orig': COLOR_ORIG, 'synth': COLOR_SYNTH}, ax=ax)

                ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

                emd_distance = categorical_emd(col_test, col_synth)
                title += ' (EMD Dist={:.3f})'.format(emd_distance)

            elif dtype == DisplayType.CONTINUOUS:
                col_test = pd.to_numeric(self.df_orig[col].dropna(), errors='coerce').dropna()
                col_synth = pd.to_numeric(self.df_synth[col].dropna(), errors='coerce').dropna()

                col_test = col_test.sample(min(sample_size, len(col_test)))
                col_synth = col_synth.sample(min(sample_size, len(col_synth)))

                if len(col_test) == 0 or len(col_synth) == 0:
                    continue

                percentiles = [remove_outliers * 100. / 2, 100 - remove_outliers * 100. / 2]
                start, end = np.percentile(col_test, percentiles)
                if start == end:
                    start, end = min(col_test), max(col_test)

                # In case the synthesized data has overflown and has much different domain
                col_synth = col_synth[(start <= col_synth) & (col_synth <= end)]

                if len(col_synth) == 0:
                    continue

                # workaround for kde failing on datasets with only one value
                if col_test.nunique() < 2 or col_synth.nunique() < 2:
                    kde = False
                    kde_kws = None
                else:
                    kde = True
                    kde_kws = {'clip': (start, end)}

                try:
                    sns.distplot(col_test, color=COLOR_ORIG, label='orig', kde=kde, kde_kws=kde_kws,
                                 hist_kws={'color': COLOR_ORIG, 'range': [start, end]}, ax=ax)
                    sns.distplot(col_synth, color=COLOR_SYNTH, label='synth', kde=kde, kde_kws=kde_kws,
                                 hist_kws={'color': COLOR_SYNTH, 'range': [start, end]}, ax=ax)
                except Exception as e:
                    print('ERROR :: Column {} cant be shown :: {}'.format(col, e))

                ks_distance = ks_2samp(col_test, col_synth)[0]
                title += ' (KS Dist={:.3f})'.format(ks_distance)

            ax.set_title(title)
            plt.legend()
        plt.title('Distributions')
        plt.show()

    def show_auto_associations(self, figsize: Tuple[float, float] = (14, 50), cols: int = 2, max_order=30):
        fig = plt.figure(figsize=figsize)
        for i, (col, dtype) in enumerate(self.display_types.items()):
            ax = fig.add_subplot(len(self.display_types), cols, i + 1)
            n = list(range(1, max_order + 1))
            original_auto = eval_metrics.calculate_auto_association(dataset=self.df_test, col=col, max_order=30)
            synth_auto = eval_metrics.calculate_auto_association(dataset=self.df_synth, col=col, max_order=30)
            ax.stem(n, original_auto, 'b', markerfmt='bo', label="Original")
            ax.stem(n, synth_auto, 'g', markerfmt='go', label="Synthetic")
            ax.set_title(label=col)
            ax.legend()

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

        self.df_orig_encoded = self.df_orig_encoded.copy().dropna()
        self.df_synth_encoded = self.df_synth_encoded.copy().dropna()
        self.df_test_encoded = self.df_test_encoded.copy().dropna()

        X, y = self.df_orig_encoded.drop(target, 1), self.df_orig_encoded[target]
        X_synth, y_synth = self.df_synth_encoded.drop(target, 1), self.df_synth_encoded[target]
        X_test, y_test = self.df_test_encoded.drop(target, 1), self.df_test_encoded[target]

        X = skip_sampling(X)
        X_synth = skip_sampling(X_synth)
        X_test = skip_sampling(X_test)

        if self.display_types[target] == DisplayType.CATEGORICAL:
            clf = clone(classifier)
            clf.fit(X, y)
            y_pred_orig = clf.predict(X_test)
            orig_score = roc_auc_score(y_test, y_pred_orig)

            clf = clone(classifier)
            clf.fit(X_synth, y_synth)
            y_pred_synth = clf.predict(X_test)
            synth_score = roc_auc_score(y_test, y_pred_synth)

            print("ROC AUC (orig):", orig_score)
            print("ROC AUC (synth):", synth_score)

        else:
            clf = clone(regressor)
            clf.fit(X, y)
            y_pred_orig = clf.predict(X_test)
            orig_score = r2_score(y_test, y_pred_orig)

            clf = clone(regressor)
            clf.fit(X_synth, y_synth)
            y_pred_synth = clf.predict(X_test)
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

    def show_distribution_distances(self) -> Tuple[float, float]:
        """Plot a barplot with KS-distances between original and synthetic columns."""
        result = []
        for col in self.df_test.columns.values:
            col_test = pd.to_numeric(self.df_test[col], errors='coerce').dropna()
            col_synth = pd.to_numeric(self.df_synth[col], errors='coerce').dropna()
            if len(col_test) == 0 or len(col_synth) == 0:
                continue
            distance = ks_2samp(col_test, col_synth)[0]
            result.append({'column': col, 'distance': distance})

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame.from_records(result)

        ks_dist_max = df['distance'].max()
        ks_dist_avg = df['distance'].mean()

        print("Max KS distance:", ks_dist_max)
        print("Average KS distance:", ks_dist_avg)

        plt.figure(figsize=(8, int(len(df) / 2) + 2))
        g = sns.barplot(y='column', x='distance', data=df)
        g.set_xlim(0.0, 1.0)
        plt.title('KS Distances')
        plt.show()

        return ks_dist_max, ks_dist_avg

    def show_emd_distances(self) -> Tuple[float, float]:
        """Plot a barplot with EMD-distances between original and synthetic columns."""
        result = []

        for col in self.categorical_cols:
            emd_distance = categorical_emd(self.df_test[col], self.df_synth[col])
            result.append({'column': col, 'emd_distance': emd_distance})

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame.from_records(result)
        emd_dist_max = df['emd_distance'].max()
        emd_dist_avg = df['emd_distance'].mean()

        print("Max EMD distance:", emd_dist_max)
        print("Average EMD distance:", emd_dist_avg)

        plt.figure(figsize=(8, int(len(df) / 2) + 2))
        g = sns.barplot(y='column', x='emd_distance', data=df)
        g.set_xlim(0.0, 1.0)
        plt.title('EMD Distances')
        plt.show()

        return emd_dist_max, emd_dist_avg

    def show_mutual_information(self) -> Tuple[float, float]:
        # normalized_mutual_info_score

        def pairwise_attributes_mutual_information(data: pd.DataFrame) -> pd.DataFrame:
            data = data.dropna()
            sorted_columns = sorted(data.columns)
            n_columns = len(sorted_columns)
            mi_df = pd.DataFrame(np.ones((n_columns, n_columns)), columns=sorted_columns, index=sorted_columns,
                                 dtype=float)

            for j in range(n_columns):
                row = sorted_columns[j]
                for i in range(j + 1, n_columns):
                    col = sorted_columns[i]
                    mi_df.loc[row, col] = mi_df.loc[col, row] = normalized_mutual_info_score(
                        data[row], data[col], average_method='arithmetic')
            return mi_df

        max_sample_size = 25_000
        sample_size = min(len(self.df_orig), len(self.df_synth), max_sample_size)

        data_pwcorr = pairwise_attributes_mutual_information(self.df_orig.sample(sample_size))
        synth_pwcorr = pairwise_attributes_mutual_information(self.df_synth.sample(sample_size))

        if len(data_pwcorr) == 0 or len(synth_pwcorr) == 0:
            return 0., 0.

        mask = np.zeros_like(data_pwcorr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)
        sns.heatmap(data_pwcorr, mask=mask, annot=True, ax=axs[0], cmap=cmap, fmt='.2f', cbar=False)
        sns.heatmap(synth_pwcorr, mask=mask, annot=True, ax=axs[1], cmap=cmap, fmt='.2f', cbar=False)

        axs[0].set_title('Original')
        axs[1].set_title('Synthesized')

        for ax in axs:
            ax.set_ylim(ax.get_ylim()[0] + .5, ax.get_ylim()[1] - .5)

        plt.tight_layout()
        plt.show()

        pw_diff = abs(data_pwcorr - synth_pwcorr)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(pw_diff, mask=mask, annot=True, vmin=0.0, vmax=1.0, cmap=cmap, fmt='.2f')
        ax.set_ylim(ax.get_ylim()[0] + .5, ax.get_ylim()[1] - .5)
        plt.tight_layout()

        pw_dist_max = np.max(np.max(pw_diff))
        pw_dist_avg = np.mean(np.mean(pw_diff))
        print("Max PW Mutual Information distance: ", pw_dist_max)
        print("Average PW Mutual Information distance: ", pw_dist_avg)

        plt.show()

        return pw_dist_max, pw_dist_avg

    def get_avg_fn(self, df, col, unique_ids: List, fn, nlags=40):
        distance = []
        if len(unique_ids) > nlags:
            for i in unique_ids:
                col_test = pd.to_numeric(df.loc[df[self.identifier] == i, col], errors='coerce').dropna().values
                if len(col_test) > 1:
                    distance.append(np.mean(fn(col_test, nlags=nlags)))
        else:
            col_test = pd.to_numeric(df[col], errors='coerce').dropna().values
            if len(col_test) > nlags:
                distance.append(np.mean(fn(col_test, nlags=nlags)))

        return distance

    def show_autocorrelation_distances(self, nlags: int = 40, plot_results: bool = True):
        """Plot a barplot with ACF-distances between original and synthetic columns."""
        result = []
        for col in self.continuous_cols:

            acf_distance_orig = self.get_avg_fn(self.df_test, col, unique_ids=self.unique_ids_orig, fn=acf, nlags=nlags)
            acf_distance_synth = self.get_avg_fn(self.df_synth, col, unique_ids=self.unique_ids_synth, fn=acf,
                                                 nlags=nlags)

            if len(acf_distance_synth) == 0 or len(acf_distance_synth) == 0:
                continue

            acf_distance = np.abs(np.mean(acf_distance_orig) - np.mean(acf_distance_synth))
            result.append({'column': col, 'distance': acf_distance})

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame.from_records(result)

        acf_dist_max = df['distance'].max()
        acf_dist_avg = df['distance'].mean()

        print("Max ACF distance:", acf_dist_max)
        print("Average ACF distance:", acf_dist_avg)

        if plot_results:
            plt.figure(figsize=(8, np.ceil(len(df) / 2)))
            g = sns.barplot(y='column', x='distance', data=df)
            g.set_xlim(0.0, 1.0)
            plt.title('ACF Distances')
            plt.show()

        return acf_dist_max, acf_dist_avg

    def show_partial_autocorrelation_distances(self, nlags=40):
        """Plot a barplot with PACF-distances between original and synthetic columns."""
        result = []
        for col in self.continuous_cols:

            pacf_distance_orig = self.get_avg_fn(self.df_test, col, unique_ids=self.unique_ids_orig, fn=pacf,
                                                 nlags=nlags)
            pacf_distance_synth = self.get_avg_fn(self.df_synth, col, unique_ids=self.unique_ids_synth, fn=pacf,
                                                  nlags=nlags)

            if len(pacf_distance_synth) == 0 or len(pacf_distance_synth) == 0:
                continue

            pacf_distance = np.abs(np.mean(pacf_distance_orig) - np.mean(pacf_distance_synth))
            result.append({'column': col, 'distance': pacf_distance})

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame.from_records(result)

        pacf_dist_max = df['distance'].max()
        pacf_dist_avg = df['distance'].mean()

        print("Max PACF distance:", pacf_dist_max)
        print("Average PACF distance:", pacf_dist_avg)

        plt.figure(figsize=(8, np.ceil(len(df) / 2)))
        g = sns.barplot(y='column', x='distance', data=df)
        g.set_xlim(0.0, 1.0)
        plt.title('PACF Distances')
        plt.show()

        return pacf_dist_max, pacf_dist_avg

    def show_series(self, num_series=10, figsize: Tuple[float, float] = None, share_axis=False, share_ids=False):

        if not figsize:
            figsize = (14, 10 * len(self.display_types))

        if self.identifier:
            num_series = min(num_series, len(np.unique(self.unique_ids_orig)), len(np.unique(self.unique_ids_synth)))
            identifiers_orig = np.random.choice(self.unique_ids_orig, num_series, replace=False)
            if share_ids:
                identifiers_synth = identifiers_orig
            else:
                identifiers_synth = np.random.choice(self.unique_ids_synth, num_series, replace=False)

        fig = plt.figure(figsize=figsize)
        for i in range(len(self.continuous_cols)):
            col = self.continuous_cols[i]

            # Original
            ax = fig.add_subplot(2 * len(self.continuous_cols), 2, 2 * i + 1)

            if self.identifier:
                for idf in identifiers_orig:
                    x = pd.to_numeric(self.df_orig.loc[self.df_orig[self.identifier] == idf, col], errors='coerce'
                                      ).dropna().values
                    if len(x) > 1:
                        ax.plot(range(len(x)), x, label=idf)
                ax.legend()
            else:
                x = pd.to_numeric(self.df_orig[col], errors='coerce').dropna().values
                if len(x) > 1:
                    ax.plot(range(len(x)), x)
            ax.set_title(col + ' (Original)')

            # Synthesized
            if share_axis:
                ax = fig.add_subplot(2 * len(self.continuous_cols), 2, 2 * i + 2, sharex=ax, sharey=ax)
            else:
                ax = fig.add_subplot(2 * len(self.continuous_cols), 2, 2 * i + 2)
            if self.identifier:
                for idf in identifiers_synth:
                    x = self.df_synth.loc[self.df_synth[self.identifier] == idf, col].dropna().values
                    ax.plot(range(len(x)), x, label=idf)
            else:
                x = self.df_synth[col].dropna().values
                ax.plot(range(len(x)), x)
            ax.legend()
            ax.set_title(col + ' (Synthesized)')

        plt.tight_layout()
        plt.show()

    def show_transition_distances(self, plot_results: bool = True):
        """Plot a barplot with ACF-distances between original and synthetic columns."""
        result = []
        for col in self.categorical_cols:
            val2idx = {v: i for i, v in enumerate(np.unique(self.df_test[col]))}
            # ORIGINAL DATA
            t_orig = np.zeros((len(val2idx), len(val2idx)))
            k = 0
            if self.identifier:
                for i in self.unique_ids_orig:
                    t_orig += eval_metrics.transition_matrix(
                        self.df_test.loc[self.df_test[self.identifier] == i, col], val2idx=val2idx)[0]
                    k += 1
                t_orig /= k
            else:
                t_orig += eval_metrics.transition_matrix(
                    self.df_test[col], val2idx=val2idx)[0]

            # Convert to dataframe
            t_orig = pd.DataFrame(t_orig, columns=list(np.unique(self.df_test[col])))
            t_orig.index = np.unique(self.df_test[col])

            # SYNTHESIZED DATA
            t_synth = np.zeros((len(val2idx), len(val2idx)))
            k = 0
            if self.identifier:
                for i in self.unique_ids_synth:
                    t_synth += eval_metrics.transition_matrix(
                        self.df_synth.loc[self.df_synth[self.identifier] == i, col], val2idx=val2idx)[0]
                    k += 1
                t_synth /= k
            else:
                t_synth += eval_metrics.transition_matrix(
                    self.df_synth[col], val2idx=val2idx)[0]

            # Convert to dataframe
            t_synth = pd.DataFrame(t_synth, columns=list(np.unique(self.df_synth[col])))
            t_synth.index = np.unique(self.df_synth[col])

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            if plot_results:
                f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7), sharex=True, sharey=True)

                # Draw the heatmap with the mask and correct aspect ratio
                sns.heatmap(t_orig, cmap=cmap, vmin=0.0, vmax=1.0, center=0,
                            square=True, linewidths=.5, cbar=False, ax=ax1, annot=True, fmt='.2f')
                sns.heatmap(t_synth, cmap=cmap, vmin=0.0, vmax=1.0, center=0,
                            square=True, linewidths=.5, cbar=False, ax=ax2, annot=True, fmt='.2f')
                sns.heatmap(abs(t_orig - t_synth), cmap=cmap, vmin=0.0, vmax=1.0, center=0,
                            square=True, linewidths=.5, cbar=False, ax=ax3, annot=True, fmt='.2f')
                ax2.set_ylim(ax2.get_ylim()[0] + .5, ax2.get_ylim()[1] - .5)

                ax1.set_title(col + ' - Transition Distances (Original)')
                ax2.set_title(col + ' - Transition Distances (Synthesized)')
                ax3.set_title(col + ' - Transition Distances (Difference)')
                plt.show()

            result.append({'column': col, 'distance': abs(t_orig - t_synth).mean().mean()})

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame.from_records(result)

        dist_max = df['distance'].max()
        dist_avg = df['distance'].mean()

        print("Max Transition distance:", dist_max)
        print("Average Transition distance:", dist_avg)

        if plot_results:
            plt.figure(figsize=(8, np.ceil(len(df) / 2)))
            g = sns.barplot(y='column', x='distance', data=df)
            g.set_xlim(0.0, 1.0)
            plt.title('ACF Distances')
            plt.show()

        return dist_max, dist_avg
