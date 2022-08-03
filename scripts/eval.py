import insight.database.db_connection as connection
import insight.database.schema as model
import insight.database.utils as util
import insight.metrics as mt

DATA_DIR = "https://raw.githubusercontent.com/synthesized-io/datasets/master"
DATASET = "tabular/templates/credit.csv"
URL = f"{DATA_DIR}/{DATASET}"

df = util.get_df(f"{DATA_DIR}/{DATASET}")
df = df.dropna()
df.name = 'credit_1'

metric = mt.TwoColumnMap(mt.KullbackLeiblerDivergence())
value = metric(df, df, session=connection.Session(expire_on_commit=False))
