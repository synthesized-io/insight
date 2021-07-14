import pandas as pd
import pytest

from synthesized import MetaExtractor
from synthesized.complex import TwoTableSynthesizer


def test_two_table_join():

    df_1 = pd.DataFrame({
        'key_1': [0, 1, 2, 3, 4, 5],
        'attr_1': [0, 1, 0, 1, 0, 1]
        })
    df_2 = pd.DataFrame({
        'key_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'attr_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
        'fk_key_1': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        })

    df_metas = (MetaExtractor.extract(df_1), MetaExtractor.extract(df_2))
    ttsyn = TwoTableSynthesizer(df_metas=df_metas, keys=('key_1', 'key_1'), relation={'key_1': 'fk_key_1'})

    df_1_fk_joined, df_2_fk_joined = ttsyn._build_relation((df_1, df_2))
    assert 'fk_key_2' not in df_2
    assert all(df_1_fk_joined.loc[df_1_fk_joined['key_1'] == 5, 'fk_count'] == 0)
    assert all(df_1_fk_joined.loc[df_1_fk_joined['key_1'] != 5, 'fk_count'] == 2)

    pd.testing.assert_frame_equal(df_1_fk_joined[['key_1', 'attr_1']], df_1)
    pd.testing.assert_frame_equal(df_2_fk_joined, df_2[['key_1', 'attr_2']])


@pytest.mark.slow
def test_two_table_synth():

    df_cust = pd.read_csv('data/two-table-customer.csv')
    df_tran = pd.read_csv('data/two-table-transaction.csv')

    dfs = (df_cust, df_tran)
    df_metas = (MetaExtractor.extract(df_cust), MetaExtractor.extract(df_tran))

    ttsyn = TwoTableSynthesizer(df_metas=df_metas, keys=('customer_id', 'transaction_id'))

    ttsyn.learn(df_train=dfs, num_iterations=100)

    df_a, df_b = ttsyn.synthesize(num_rows=950)
    assert len(df_a.merge(df_b, on='customer_id', how='left')) >= len(df_cust)


def test_repr():
    df_cust = pd.read_csv('data/two-table-customer.csv')
    df_tran = pd.read_csv('data/two-table-transaction.csv')

    dfs = (df_cust, df_tran)
    df_metas = (MetaExtractor.extract(df_cust), MetaExtractor.extract(df_tran))

    ttsyn = TwoTableSynthesizer(df_metas=df_metas, keys=('customer_id', 'transaction_id'))
    assert repr(ttsyn) == f"TwoTableSynthesizer(df_metas={repr(df_metas)}, keys=('customer_id', 'transaction_id'), relation={{'customer_id': 'customer_id'}})"


@pytest.mark.slow
def test_two_table_synth_fit_sample():

    df_cust = pd.read_csv('data/two-table-customer.csv')
    df_tran = pd.read_csv('data/two-table-transaction.csv')

    dfs = (df_cust, df_tran)
    df_metas = (MetaExtractor.extract(df_cust), MetaExtractor.extract(df_tran))

    ttsyn = TwoTableSynthesizer(df_metas=df_metas, keys=('customer_id', 'transaction_id'))

    ttsyn.fit(df_train=dfs, num_iterations=100)

    df_a, df_b = ttsyn.sample(num_rows=950)
    assert len(df_a.merge(df_b, on='customer_id', how='left')) >= len(df_cust)
