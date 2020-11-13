import pandas as pd


class Relation:

    def __init__(self, name: str):
        self.name = name

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
