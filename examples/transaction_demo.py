import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def clean_dataset(df):
    df = df[["account_id", "date", "operation", "amount"]]
    df = df.dropna()
    df['operation'] = df['operation'].astype(int)
    return df


class TransactionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._dates = None
        self._categories = None

    def fit(self, X, y=None):
        dates = X['date'].unique()
        dates = np.sort(dates)

        categories = X['operation'].unique()
        categories = np.sort(categories)

        self._dates = dates
        self._categories = categories
        return self

    def transform(self, X):
        def to_vector(group):
            res = np.zeros(len(self._categories) * len(self._dates), group['amount'].dtype)
            for _, row in group.iterrows():
                category_idx = np.where(self._categories == row['operation'])[0][0]
                date_idx = np.where(self._dates == row['date'])[0][0]
                idx = date_idx * len(self._categories) + category_idx
                res[idx] = row['amount']
            return pd.Series(res)
        return X.groupby(by='account_id', sort=False, as_index=False).apply(to_vector)

    def inverse_transform(self, X):
        def to_transactions(row):
            account_id = np.random.randint(sys.maxsize)
            rows = []
            for idx, amount in row.iteritems():
                if amount == 0.0:
                    continue
                category_idx = idx % len(self._categories)
                date_idx = int((idx - category_idx) / len(self._categories))
                category = self._categories[category_idx]
                date = self._dates[date_idx]
                rows.append({'account_id': account_id, 'date': date, 'operation': category, 'amount': amount})
            return pd.DataFrame.from_records(rows, columns=['account_id', 'date', 'operation', 'amount'])
        dfs = []
        for _, row in pd.DataFrame(X).iterrows():
            dfs.append(to_transactions(row))
        return pd.concat(dfs)
