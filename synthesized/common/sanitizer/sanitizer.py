import pandas as pd


class Sanitizer:
    """This interface defines a sanitization method of synthetic data."""

    def sanitize(self, df_synthesized: pd.DataFrame) -> pd.DataFrame:
        pass


class DefaultSanitizer(Sanitizer):
    """The default implementation. Drops duplicates. Floats are rounded."""

    FLOAT_DECIMAL = 5

    def __init__(self, df_original: pd.DataFrame) -> None:
        self.df_original = df_original

    def sanitize(self, df_synthesized: pd.DataFrame) -> pd.DataFrame:
        """Drop rows in df_synthesized that are present in df_original."""

        def normalize_tuple(nt):
            res = []
            for field in nt:
                if isinstance(field, float):
                    field = round(field, DefaultSanitizer.FLOAT_DECIMAL)
                res.append(field)
            return tuple(res)

        original_rows = {normalize_tuple(row) for row in self.df_original.itertuples(index=False)}
        to_drop = []
        for i, row in enumerate(df_synthesized.itertuples(index=False)):
            if normalize_tuple(row) in original_rows:
                to_drop.append(i)
        return df_synthesized.drop(to_drop)
