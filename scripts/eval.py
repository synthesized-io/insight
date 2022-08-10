import synthesized_datasets as datasets

import insight.database.db_connection as connection
import insight.metrics as mt

data = datasets.CREDIT.credit
df = data.load()
df = df.dropna()
df.name = data.name

metric = mt.TwoColumnMap(mt.KullbackLeiblerDivergence())
value = metric(df, df, session=connection.Session(expire_on_commit=False))
