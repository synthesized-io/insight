import pandas as pd

from .value import Value


class EnumerationValue(Value):

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=self.name, axis=1)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        df.loc[:, self.name] = df.index + 1
        return df
