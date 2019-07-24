# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# The testing environment

from __future__ import division, print_function, absolute_import

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyemd.emd import emd_samples
from scipy.stats import ks_2samp, pearsonr
from sklearn import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.metrics.scorer import roc_auc_scorer
from sklearn.preprocessing import StandardScaler

from synthesized.common.values import CategoricalValue
from synthesized.common.values import ContinuousValue
from synthesized.common.values import SamplingValue
from .util import categorical_emd

MAX_CATEGORIES = 50
COLOR_ORIG = "#00AB26"
COLOR_SYNTH = "#2794F3"


class DisplayType(Enum):
    CATEGORICAL = 1
    CATEGORICAL_SIMILARITY = 2
    CONTINUOUS = 3


class UtilityTesting:
    def __init__(self, synthesizer, df_orig, df_test, df_synth):
        self.display_types = UtilityTesting._detect_display_types(synthesizer)
        self.value_by_name = {}
        for v in synthesizer.values:
            self.value_by_name[v.name] = v
        self.df_orig = df_orig
        self.df_test = df_test
        self.df_synth = df_synth

        self.df_orig_encoded = synthesizer.preprocess(df=df_orig)
        self.df_test_encoded = synthesizer.preprocess(df=df_test)
        self.df_synth_encoded = synthesizer.preprocess(df=df_synth)

    @staticmethod
    def _detect_display_types(synthesizer):
        result = {}
        for value in synthesizer.values:
            if isinstance(value, ContinuousValue):
                result[value.name] = DisplayType.CONTINUOUS
            elif isinstance(value, CategoricalValue):
                result[value.name] = DisplayType.CATEGORICAL
        return result

    def show_corr_matrices(self, figsize=(15, 11)):
        def show_corr_matrix(df, title=None, ax=None):
            sns.set(style="white")

            # Compute the correlation matrix
            corr = df.corr()

            # Generate a mask for the upper triangle
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            hm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                             square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

            if title is not None:
                hm.set_title(title)

        # Set up the matplotlib figure
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        show_corr_matrix(self.df_orig, title='Original', ax=ax1)
        show_corr_matrix(self.df_synth, title='Synthetic', ax=ax2)

    def show_distributions(self, remove_outliers=0.0, figsize=(14, 50), cols=2):
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
                             hist_kws={"color": COLOR_ORIG, 'range': [start, end]}, ax=ax)
                sns.distplot(self.df_synth[col], color=COLOR_SYNTH, label='synth', kde=kde,
                             kde_kws={'clip': (start, end)},
                             hist_kws={"color": COLOR_SYNTH, 'range': [start, end]}, ax=ax)
            plt.legend()

    def improve_column(self, column, clf):
        X_columns = list(set(self.df_orig.columns.values) - {column})
        X = self.df_orig[X_columns]
        y = self.df_orig[column]
        clf.fit(X, y)
        self.df_synth[column] = clf.predict(self.df_synth[X_columns])
        return self.df_synth

    def utility(self, target, classifier=GradientBoostingClassifier(), regressor=GradientBoostingRegressor()):
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

            print('ROC AUC (orig):', orig_score)
            print('ROC AUC (synth):', synth_score)
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
            print('R2 (orig):', orig_score)
            print('R2 (synth):', synth_score)
            return synth_score

    def autocorrelation_diff_plot_seaborn(self, max_lag=100):
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

        sns.set(style="whitegrid")

        data = pd.DataFrame({"Original": y_orig, "Synthetic": y_synth})
        sns.lineplot(data=data, palette=[COLOR_SYNTH, COLOR_ORIG], linewidth=2.5)
        return mean_squared_error(y_orig, y_synth)

    @staticmethod
    def _bin_edges(a):
        _, edges = np.histogram(a, bins=10)
        return edges

    @staticmethod
    def _to_categorical(a, edges):
        return np.digitize(a, edges)

    def compare_marginal_distributions(self, target_column, conditional_column, bins=4):
        if self.df_orig[conditional_column].dtype == 'O':
            return self._compare_marginal_distributions_categorical(target_column, conditional_column)
        else:
            return self._compare_marginal_distributions_continuous(target_column, conditional_column, bins)

    def _compare_marginal_distributions_categorical(self, target_column, conditional_column):
        target_emd = '{} EMD'.format(target_column)
        result = []
        for val in np.unique(self.df_orig[conditional_column]):
            df_orig_target = self.df_orig[self.df_orig[conditional_column] == val][target_column]
            df_synth_target = self.df_synth[self.df_synth[conditional_column] == val][target_column]
            if len(df_orig_target) == 0 or len(df_synth_target) == 0:
                emd = float('inf')
            else:
                if self.df_orig[target_column].dtype.kind == 'O':
                    emd = categorical_emd(df_orig_target, df_synth_target)
                else:
                    emd = emd_samples(df_orig_target, df_synth_target)
            result.append({
                conditional_column: val,
                target_emd: emd
            })
        return pd.DataFrame.from_records(result, columns=[conditional_column, target_emd])

    def _compare_marginal_distributions_continuous(self, target_column, conditional_column, bins):
        _, edges = np.histogram(self.df_orig[conditional_column], bins=bins)
        target_emd = '{} EMD'.format(target_column)
        result = []
        for i in range(0, len(edges) - 1):
            df_orig_target = self.df_orig[
                (self.df_orig[conditional_column] >= edges[i]) & (self.df_orig[conditional_column] < edges[i + 1])][
                target_column]
            df_synth_target = self.df_synth[
                (self.df_synth[conditional_column] >= edges[i]) & (self.df_synth[conditional_column] < edges[i + 1])][
                target_column]
            if len(df_orig_target) == 0 or len(df_synth_target) == 0:
                emd = float('inf')
            else:
                if self.df_orig[target_column].dtype.kind == 'O':
                    emd = categorical_emd(df_orig_target, df_synth_target)
                else:
                    emd = emd_samples(df_orig_target, df_synth_target)
            result.append({
                conditional_column: '[{}, {})'.format(edges[i], edges[i + 1]),
                target_emd: emd
            })
        return pd.DataFrame.from_records(result, columns=[conditional_column, target_emd])

    def show_distribution_distances(self):
        result = []
        for col in self.df_test.columns.values:
            distance = ks_2samp(self.df_test[col], self.df_synth[col])[0]
            result.append({'column': col, 'distance': distance})
        df = pd.DataFrame.from_records(result)
        print('Average distance:', df['distance'].mean())
        g = sns.barplot(y='column', x='distance', data=df)
        g.set_xlim(0.0, 1.0)

    def show_correlation_diffs(self, threshold=0.0, report=False):
        result = []
        cols = list(self.df_orig.columns.values)
        cols = list(filter(lambda col: self.df_orig.dtypes[col].kind != 'O', cols))
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i < j:
                    corr1 = pearsonr(self.df_orig[cols[i]], self.df_orig[cols[j]])[0]
                    corr2 = pearsonr(self.df_synth[cols[i]], self.df_synth[cols[j]])[0]
                    if math.isnan(corr1) or math.isnan(corr2):
                        continue
                    result.append({'pair': cols[i] + ' / ' + cols[j], 'diff': corr2 - corr1})
        df = pd.DataFrame.from_records(result)
        print('Average diff:', df['diff'].mean())
        df = df[np.abs(df['diff']) > threshold]
        df = df.sort_values(by='diff', ascending=False)
        corrs = [i["diff"] for i in result]
        sns.distplot(corrs, color=COLOR_ORIG, label='Correlations Difference', kde=False, hist=True)
        if report:
            return df

    def debug_column(self, col):
        start, end = np.percentile(self.df_orig[col], [2.5, 97.5])  # TODO parametrize
        plt.hist([self.df_orig[col], self.df_synth[col]], label=['orig', 'synth'], range=(start, end))
        plt.legend()
