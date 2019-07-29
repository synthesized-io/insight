import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from synthesized.highdim import HighDimSynthesizer
from synthesized.common.values.identify_value import identify_value
from synthesized.common.values.identify_rules import identify_rules

import os
import time

BASEDIR = os.path.dirname(__file__)


def main():
    df = pd.read_csv(os.path.join(BASEDIR, 'data', 'credit_retired.csv'))
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    with HighDimSynthesizer(df=train, find_rules='all') as synthesizer:
        synthesizer.learn(df_train=train, num_iterations=200)
        synthesized = synthesizer.synthesize(num_rows=len(test))

    assert (synthesized['age'] >= 65).sum() == synthesized['retired'].sum()


def stress_test():
    df = pd.DataFrame(np.random.randn(10**6, 100))
    # Make lots of categorical variables
    df.iloc[:, 50:80] = (df.iloc[:, 50:80] > 0)
    df.iloc[:, 80:98] = -1 * (df.iloc[:, 80:98] < -1) + 1 * (df.iloc[:, 80:98] > 1)
    df.iloc[:, 99] = (df.iloc[:,98] > 0)
    df.columns = [str(x) for x in df.columns]

    dummy = HighDimSynthesizer(df=pd.DataFrame(np.random.randn(5, 5)))

    # Make the list of values
    start1 = time.time()
    values = list()
    for name in df.columns:
        value = identify_value(module=dummy, df=df[name], name=name)
        assert len(value.columns()) == 1 and value.columns()[0] == name
        values.append(value)

    # Automatic extraction of specification parameters
    df = df.copy()
    for value in values:
        value.extract(df=df)

    # Identify deterministic rules
    start = time.time()
    values = identify_rules(values=values, df=df, tests='all')
    print('That took {:.3e}s'.format(time.time()-start))
    print('Overall took {:.3e}s'.format(time.time()-start1))


if __name__ == '__main__':
    #  main()
    #  stress_test_grid('find_piecewise')
    stress_test()
