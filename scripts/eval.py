import synthesized_datasets as datasets

import insight.metrics as mt

data = datasets.CREDIT.credit
df = data.load()
df = df.dropna()
df.name = data.name

metric = mt.TwoColumnMap(mt.KullbackLeiblerDivergence())
value = metric(df, df)
