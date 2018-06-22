import pandas as pd
from synthesized.core import BasicSynthesizer
# import examples.transaction_demo


data = pd.read_csv('data/transactions.csv')
# df = df[pd.to_datetime(df['date']) < pd.to_datetime('1994-01-01')]
data = data[['type', 'operation', 'amount']]
data = data.dropna()
data = data[data['type'] != 'VYBER']
data['type'] = data['type'].astype(dtype='int')
data['type'] = data['type'].astype(dtype='category')
data['operation'] = data['operation'].astype(dtype='int')
data['operation'] = data['operation'].astype(dtype='category')
data['amount'] = data['amount'].astype(dtype='float32')
print(data.head(5))

synthesizer = BasicSynthesizer(dtypes=data.dtypes, encoded_size=32)
synthesizer.fit(data=data)
synthesized = synthesizer.synthesize(n=50)
print(synthesized)
