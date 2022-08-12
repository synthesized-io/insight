import synthesized_datasets as datasets

import insight.metrics as mt

df = datasets.CREDIT.credit.load()
df = df.dropna()

metric = mt.TwoColumnMap(mt.KullbackLeiblerDivergence())
value = metric(df.sample(200), df.sample(200))

print(value)
