# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# The testing environment

from __future__ import division, print_function, absolute_import

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
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics.scorer import roc_auc_scorer
from sklearn.preprocessing import StandardScaler

from synthesized.core.values import ContinuousValue
from synthesized.core.values import CategoricalValue


from .util import categorical_emd

MAX_CATEGORIES = 50
COLOR_ORIG = "#00AB26"
COLOR_SYNTH = "#2794F3"


class DisplayType(Enum):
    CATEGORICAL = 1
    CATEGORICAL_SIMILARITY = 2
    CONTINUOUS = 3


def detect_display_types(synthesizer, df):
    result = {}
    for name, dtype in zip(df.dtypes.axes[0], df.dtypes):
        value = synthesizer.get_value(name=name, dtype=dtype, data=df)
        if isinstance(value, ContinuousValue):
            result[name] = DisplayType.CONTINUOUS
        elif isinstance(value, CategoricalValue):
            if value.similarity_based:
                result[name] = DisplayType.CATEGORICAL_SIMILARITY
            else:
                result[name] = DisplayType.CATEGORICAL
    return result


class UtilityTesting:
    def __init__(self, synthesizer, df_orig, df_test, df_synth):
        self.display_types = detect_display_types(synthesizer, df_orig)
        self.df_orig = synthesizer.preprocess(data=df_orig.copy())
        self.df_test = synthesizer.preprocess(data=df_test.copy())
        self.df_synth = synthesizer.preprocess(data=df_synth.copy())

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

    def show_distributions(self, figsize=(14, 40), cols=2):
        fig = plt.figure(figsize=figsize)
        for i, (col, dtype) in enumerate(self.display_types.items()):
            ax = fig.add_subplot(len(self.display_types), cols, i + 1)
            if dtype == DisplayType.CATEGORICAL:
                sns.distplot(self.df_test[col], color=COLOR_ORIG, label='Orig', kde=False, hist=True, norm_hist=True)
                sns.distplot(self.df_synth[col], color=COLOR_SYNTH, label='Synth', kde=False, hist=True, norm_hist=True)
                # plt.hist([self.df_test[col], self.df_synth[col]], label=['orig', 'synth'], normed=True)
                # ax.set_xlabel(col)
            elif dtype == DisplayType.CATEGORICAL_SIMILARITY:
                # workaround for kde failing on datasets with only one value
                if self.df_test[col].nunique() < 2 or self.df_synth[col].nunique() < 2:
                    kde = False
                else:
                    kde = True
                sns.distplot(self.df_test[col], color=COLOR_ORIG, label='Orig', kde=kde, hist=True, norm_hist=True,
                             hist_kws={"color": COLOR_ORIG})
                sns.distplot(self.df_synth[col], color=COLOR_SYNTH, label='Synth', kde=kde, hist=True, norm_hist=True,
                             hist_kws={"color": COLOR_SYNTH})
            elif dtype == DisplayType.CONTINUOUS:
                start, end = np.percentile(self.df_test[col], [2.5, 97.5])  # TODO parametrize
                # sns.distplot(self.df_test[col], hist=False, kde=True, label='orig', ax=ax)
                # sns.distplot(self.df_synth[col], hist=False, kde=True, label='synth', ax=ax)
                # ax.set(xlim=(start, end))
                # workaround for kde failing on datasets with only one value
                if self.df_test[col].nunique() < 2 or self.df_synth[col].nunique() < 2:
                    kde = False
                else:
                    kde = True
                sns.distplot(self.df_test[col], color=COLOR_ORIG, label='Orig', kde=kde, kde_kws={'clip': (start, end)},
                             hist_kws={"color": COLOR_ORIG, 'range': [start, end]})
                sns.distplot(self.df_synth[col], color=COLOR_SYNTH, label='Synth', kde=kde, kde_kws={'clip': (start, end)},
                             hist_kws={"color": COLOR_SYNTH, 'range': [start, end]})
                # plt.hist([self.df_test[col], self.df_synth[col]], label=['orig', 'synth'], range=(start, end), normed=True)
            plt.legend()

    def improve_column(self, column, clf):
        X_columns = list(set(self.df_orig.columns.values) - {column})
        X = self.df_orig[X_columns]
        y = self.df_orig[column]
        clf.fit(X, y)
        self.df_synth[column] = clf.predict(self.df_synth[X_columns])
        return self.df_synth

    def utility(self, target, classifier=GradientBoostingClassifier(), regressor=GradientBoostingRegressor()):
        X, y = self.df_orig.drop(target, 1), self.df_orig[target]
        X_synth, y_synth = self.df_synth.drop(target, 1), self.df_synth[target]
        X_test, y_test = self.df_test.drop(target, 1), self.df_test[target]
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

            orig_score = np.sqrt(mean_squared_error(y_test, y_pred_orig))
            synth_score = np.sqrt(mean_squared_error(y_test, y_pred_synth))
            print('RMSE (orig):', orig_score)
            print('RMSE (synth):', synth_score)
            return synth_score

    def estimate_utility(self, classifier=LogisticRegression(), regressor=LinearRegression()):
        dtypes = dict(self.display_types)
        df_orig = self.df_orig.copy()
        df_test = self.df_test.copy()
        df_synth = self.df_synth.copy()
        result = []
        columns_set = set(df_orig.columns.values)
        y_columns = sorted(list(columns_set))
        y_columns_new = []
        y_orig_columns = {}
        for i, y_column in enumerate(y_columns):
            y_columns_new.append(y_column)
            if dtypes[y_column] != DisplayType.CATEGORICAL:
                categorical_y_column = y_column + ' (categorical reduction)'
                edges = bin_edges(df_orig[y_column])
                df_orig[categorical_y_column] = to_categorical(df_orig[y_column], edges)
                df_test[categorical_y_column] = to_categorical(df_test[y_column], edges)
                df_synth[categorical_y_column] = to_categorical(df_synth[y_column], edges)
                dtypes[categorical_y_column] = DisplayType.CATEGORICAL
                y_columns_new.append(categorical_y_column)
                y_orig_columns[categorical_y_column] = y_column

        for y_column in y_columns_new:
            to_exclude = [y_column]
            if y_column in y_orig_columns:
                to_exclude.append(y_orig_columns[y_column])
            X_columns = list(columns_set.difference(to_exclude))
            X_orig_train = df_orig[X_columns]
            y_orig_train = df_orig[y_column]

            X_orig_test = df_test[X_columns]
            y_orig_test = df_test[y_column]

            X_synth = df_synth[X_columns]
            y_synth = df_synth[y_column]

            scaler = StandardScaler()
            X_orig_train = scaler.fit_transform(X_orig_train)
            X_orig_test = scaler.transform(X_orig_test)
            X_synth = scaler.transform(X_synth)

            if dtypes[y_column] == DisplayType.CATEGORICAL:
                estimator = classifier
                dummy_estimator = DummyClassifier(strategy="prior")
            else:
                estimator = regressor
                dummy_estimator = DummyRegressor()

            orig_score = max(clone(estimator).fit(X_orig_train, y_orig_train).score(X_orig_test, y_orig_test), 0.0)
            synth_score = max(clone(estimator).fit(X_synth, y_synth).score(X_orig_test, y_orig_test), 0.0)
            dummy_orig_score = max(
                clone(dummy_estimator).fit(X_orig_train, y_orig_train).score(X_orig_test, y_orig_test), 0.0)
            y_orig_pred = clone(estimator).fit(X_orig_train, y_orig_train).predict(X_orig_test)
            y_synth_pred = clone(estimator).fit(X_synth, y_synth).predict(X_orig_test)

            if dtypes[y_column] == DisplayType.CATEGORICAL:
                orig_error = 1 - accuracy_score(y_orig_test, y_orig_pred)
                synth_error = 1 - accuracy_score(y_orig_test, y_synth_pred)
            else:
                orig_error = np.sqrt(mean_squared_error(y_orig_test, y_orig_pred))
                synth_error = np.sqrt(mean_squared_error(y_orig_test, y_synth_pred))

            orig_gain = max(orig_score - dummy_orig_score, 0.0)
            synth_gain = max(synth_score - dummy_orig_score, 0.0)

            if orig_gain == 0.0:
                if synth_gain == 0.0:
                    score_utility = float("nan")
                else:
                    score_utility = 0.0
            else:
                score_utility = synth_gain / orig_gain

            if synth_error == 0.0:
                error_utility = 1.0
            else:
                error_utility = orig_error / synth_error

            result.append({
                'target_column': y_column,
                'estimator': estimator.__class__.__name__,
                'dummy_original_score': dummy_orig_score,
                'original_score': orig_score,
                'synth_score': synth_score,
                'orig_error': orig_error,
                'synth_error': synth_error,
                'score_utility': score_utility,
                'error_utility': error_utility
            })

        return pd.DataFrame.from_records(result, columns=[
            'target_column',
            'estimator',
            'dummy_original_score',
            'original_score',
            'synth_score',
            'orig_error',
            'synth_error',
            'score_utility',
            'error_utility'
        ])

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
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i < j:
                    corr1 = pearsonr(self.df_orig[cols[i]], self.df_orig[cols[j]])[0]
                    corr2 = pearsonr(self.df_synth[cols[i]], self.df_synth[cols[j]])[0]
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
        start, end = np.percentile(self.df_orig[col], [2.5, 97.5])  #TODO parametrize
        plt.hist([self.df_orig[col], self.df_synth[col]], label=['orig', 'synth'], range=(start, end))
        plt.legend()


def bin_edges(a):
    _, edges = np.histogram(a, bins=10)
    return edges


def to_categorical(a, edges):
    return np.digitize(a, edges)


# Deprecated
class TestingEnvironment:
    def __init__(self):
        self.synth_stat_models = [{"model": LogisticRegression, "variables": self.get_LogisticRegression_variables},
                                  {"model": LinearRegression, "variables": self.get_LinearRegression_variables}]

    def get_LogisticRegression_variables(self, dataset):
        X = ["operation", "amount", "balance"]
        y = ["type"]
        dataset = dataset[X + y].dropna()
        return {"X": dataset[X], "y": dataset[y]}

    def get_LinearRegression_variables(self, dataset):
        dataset['date'] = pd.to_datetime(dataset['date'])
        account_ids = dataset["account_id"].unique()
        operations = dataset["operation"].unique()
        X = np.array([]).reshape(0, 5)
        y = np.array([])
        for account_id in account_ids:
            user = dataset[dataset["account_id"] == account_id]
            for month in range(1, 12):
                X_local = np.array([])
                user_local = user[user["date"].dt.month == month]
                for operation in operations:
                    X_local = np.append(X_local, user_local[user_local["operation"] == operation]["amount"].sum())
                X = np.vstack((X, X_local))
                next_month_df = user[(user["date"].dt.month == month + 1) & (user["operation"] == 2)]
                y = np.concatenate((y, np.array([next_month_df["amount"].sum()])))
        return {"X": X, "y": y}

    def compare_empirical_densities(self, original_dataset, synthetic_dataset, target_field):
        import plotly.figure_factory as ff
        from plotly.offline import init_notebook_mode, iplot

        init_notebook_mode()  # distribution of spending on the first month in the first category

        assert (len(synthetic_dataset[0]) > target_field), "Field value is out of range for the synthesized data"
        assert (len(original_dataset[0]) > target_field), "Field value is out of range for the real data"

        fake_spendings = np.asarray([a[target_field] for a in synthetic_dataset])
        true_spendings = np.asarray([a[target_field] for a in original_dataset])

        fake_spendings = np.asarray([a for a in fake_spendings if a != 0])
        true_spendings = np.asarray([a for a in true_spendings if a != 0])

        print("The sum of values in the field %s of synthesized data is %s" % (target_field, sum(fake_spendings)))
        print("The sum of values in the field %s of original data is %s" % (target_field, sum(true_spendings)))

        assert (sum(fake_spendings) > 0), "zero value in the field in the synthesized dataset"
        assert (sum(true_spendings) > 0), "zero value in the field in the read dataset"

        init_notebook_mode(connected=True)
        # Group data together
        hist_data = [true_spendings, fake_spendings]

        group_labels = ['Real data', 'Synthetic data']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.01, .01])

        # Plot!
        iplot(fig, filename='jupyter/basic_bar')

    def get_the_closest_element(self, data_set, element):
        minimal_distance = 100
        the_closest_fake_user = 0
        for user in data_set:
            if np.sum(abs(element - user)) < minimal_distance:
                the_closest_fake_user = user
                minimal_distance = np.sum(abs(element - user))
        return (the_closest_fake_user, minimal_distance)

    def get_reconstruction_errors(self, original_dataset, original_dataset_batch, synthetic_dataset):
        import plotly.figure_factory as ff
        from plotly.offline import init_notebook_mode, iplot

        init_notebook_mode()  # distribution of spending on the first month in the first category

        init_notebook_mode(connected=True)

        errors_synthetic = [self.get_the_closest_element(synthetic_dataset, original_dataset_batch[i])[1] for i in
                            range(0, len(original_dataset_batch))]
        errors_original = [self.get_the_closest_element(original_dataset, original_dataset_batch[i])[1] for i in
                           range(0, len(original_dataset_batch))]

        # Group data together
        # errors_scaled = [error/31 for error in errors]
        hist_data = [errors_original, errors_synthetic]

        group_labels = ['Errors for real', 'Errors for synthetic']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.001, .001])

        # Plot!
        iplot(fig, filename='jupyter/basic_bar')
