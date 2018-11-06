import sys
import warnings
from math import isclose

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import calendar
import datetime


def clean_dataset(df):
    df = df[["account_id", "date", "operation", "amount"]]
    df = df.dropna()
    df['operation'] = df['operation'].astype(int)
    return df


def segment_by_month(df):
    df = df.copy()
    df['day'] = pd.to_datetime(df['date']).dt.day
    df['segment_id'] = df['account_id'].apply(str) + ':' + pd.to_datetime(df['date']).dt.to_period('M').apply(str)
    df.drop(columns=['account_id'], inplace=True)
    return df


def reconstruct_dates(df, year):
    df = df.copy()

    def reconstruct_date(row):
        month = 1 + int(row['segment_id']) % 12
        day = int(row['day'])
        last_day = calendar.monthrange(year, month)[1]
        day = min(day, last_day)
        return datetime.datetime(year, month, day)

    df['date'] = df.apply(reconstruct_date, axis=1)
    df.drop(columns=['day'], inplace=True)
    return df


def merge_segments(df, segments_per_account=20):
    df = df.copy()
    num_accounts = int(1. * len(df) / segments_per_account)
    df['account_id'] = df['segment_id'] % num_accounts
    df.drop(columns=['segment_id'], inplace=True)
    return df


class TransactionVectorizer(BaseEstimator, TransformerMixin):
    """Groups transactions by group_column and flattens dim1_column and dim2_column
    into one-dimensional array of length dim1_size * dim2_size"""

    def __init__(self, group_column, value_column, dim1_column, dim2_column, dim1_size, dim2_size):
        self.group_column = group_column
        self.value_column = value_column
        self.dim1_column = dim1_column
        self.dim2_column = dim2_column
        self.dim1_size = dim1_size
        self.dim2_size = dim2_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def to_vector(group):
            res = np.zeros(self.dim1_size * self.dim2_size, group[self.value_column].dtype)
            for _, row in group.iterrows():
                dim1 = int(row[self.dim1_column])
                dim2 = int(row[self.dim2_column])
                idx = dim1 * self.dim2_size + dim2
                res[idx] = row[self.value_column]
            return pd.Series(res)

        return X.groupby(by=self.group_column, sort=True, as_index=False).apply(to_vector)

    def inverse_transform(self, X):
        def to_transactions(row):
            group_value = np.random.randint(sys.maxsize)
            rows = []
            for idx, value in row.iteritems():
                if isclose(value, 0.0, abs_tol=1e-5):
                    continue
                dim2 = int(idx % self.dim2_size)
                dim1 = int((idx - dim2) / self.dim2_size)
                rows.append({self.group_column: group_value, self.dim1_column: dim1, self.dim2_column: dim2,
                             self.value_column: round(value, 2)})
            return pd.DataFrame.from_records(rows, columns=[self.group_column, self.dim1_column, self.dim2_column,
                                                            self.value_column])

        dfs = []
        for _, row in pd.DataFrame(X).iterrows():
            df = to_transactions(row)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)


def suppress_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


@suppress_warnings
def show_data(array_a, array_b=None, array_c=None, nrow=2, ncol=10, figsize=None, save_loc=None):
    # import without warnings
    from matplotlib import pyplot as plt

    # if both are None, just plot one
    if array_b is None and array_c is None:
        nrow = 1

    # if kw specifically makes B None, shift it over
    elif array_b is None:
        array_b = array_c
        array_c = None
        nrow = 2

    # otherwise if just plotting the first two...
    elif array_c is None:
        nrow = 2

    elif array_b is not None and array_c is not None:
        nrow = 3

    if nrow not in (1, 2, 3):
        raise ValueError('nrow must be in (1, 2)')

    if figsize is None:
        figsize = (ncol, nrow)

    f, a = plt.subplots(nrow, ncol, figsize=figsize)
    arrays = [array_a, array_b, array_c]

    def _do_show(the_figure, the_array):
        the_figure.imshow(the_array)
        the_figure.axis('off')

    for i in range(ncol):
        if nrow > 1:
            for j in range(nrow):
                _do_show(a[j][i], np.reshape(arrays[j][i], (16, 8)))
        else:
            _do_show(a[i], np.reshape(array_a[i], (16, 8)))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    f.show()
    plt.draw()

    # if save...
    if save_loc is not None:
        plt.savefig(save_loc)
