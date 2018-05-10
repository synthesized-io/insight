import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def clean_dataset(df):
    df = df[["account_id", "date", "operation", "amount"]]
    df = df.dropna()
    df['operation'] = df['operation'].astype(int)
    return df


def segment_by_month(df):
    df = df.copy()
    df['day'] = pd.to_datetime(df['date']).dt.day
    df['segment_id'] = df['account_id'].apply(str) + ':' + pd.to_datetime(df['date']).dt.to_period('M').apply(str)
    df.drop(columns=['date', 'account_id'], inplace=True)
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
                idx = dim1 * self.dim1_size + dim2
                res[idx] = row[self.value_column]
            return pd.Series(res)

        return X.groupby(by=self.group_column, sort=False, as_index=False).apply(to_vector)

    def inverse_transform(self, X):
        def to_transactions(row):
            group_value = np.random.randint(sys.maxsize)
            rows = []
            for idx, value in row.iteritems():
                if value == 0.0:
                    continue
                dim1 = idx % self.dim1_size
                dim2 = int((idx - dim1) / self.dim1_size)
                rows.append({self.group_column: group_value, self.dim1_column: dim1, self.dim2_column: dim2,
                             self.value_column: value})
            return pd.DataFrame.from_records(rows, columns=[self.group_column, self.dim2_column, self.dim1_column,
                                                            self.value_column])

        dfs = []
        for _, row in pd.DataFrame(X).iterrows():
            dfs.append(to_transactions(row))
        return pd.concat(dfs, ignore_index=True)
