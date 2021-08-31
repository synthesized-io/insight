from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Check(ABC):
    """Abstract class for checking column types
    Implement this class for the customized logic for different types of checks
    """
    @abstractmethod
    def continuous(self, sr: pd.Series):
        """Checks if the given series is continuous"""
        pass

    @abstractmethod
    def categorical(self, sr: pd.Series):
        """Checks if the given series is categorical"""
        pass

    @abstractmethod
    def ordinal(self, sr: pd.Series):
        """Checks if the given series is ordinal
        A series whose values can be ordered is ordinal
        """
        pass

    @abstractmethod
    def affine(self, sr: pd.Series):
        """Checks if the given series is affine"""
        pass

    @abstractmethod
    def same_domain(self, sr_a: pd.Series, sr_b: pd.Series):
        """Checks if the given columns have the same domain of values"""
        pass


class ColumnCheck(Check):
    """Utility to check the column or a pairs of columns for different conditions
    Args:
        min_num_unique (int, optional):
            Minimum number of unique values for the data to be continuous.
            Default value is 10.
        ctl_mult (float, optional):
            Categorical threshold log multiplier.
            Default value is 2.5.
    """
    def __init__(self,
                 min_num_unique: int = 10,
                 ctl_mult: float = 2.5):
        self.min_num_unique = min_num_unique
        self.ctl_mult = ctl_mult

    def infer_dtype(self, sr: pd.Series) -> pd.Series:
        """Infers the type of the data and converts the data to it.
        Args:
            sr (str):
                The column of the dataframe to transform.
        Returns:
            pd.Series:
                The column converted to its inferred type.
        """

        col = sr.copy()

        in_dtype = str(col.dtype)
        n_nans = col.isna().sum()
        n_empty_str = 0 if not pd.api.types.is_string_dtype(col) else (sr == "").sum()

        # Try to convert it to numeric
        if col.dtype.kind not in ("i", "u", "f") and col.dtype.kind != 'M':
            col_num = pd.to_numeric(col, errors="coerce")
            if col_num.isna().sum() == n_nans + n_empty_str:
                col = col_num

        # Try to convert it to date
        if col.dtype.kind == "O" or col.dtype.kind == 'M':
            try:
                col_date = pd.to_datetime(col, errors="coerce")
            except TypeError:
                col_date = pd.to_datetime(col.astype(str), errors="coerce")
            if col_date.isna().sum() == n_nans + n_empty_str:
                col = col_date

        out_dtype = str(col.dtype)

        if out_dtype == in_dtype:
            return col
        elif out_dtype in ("i", "u", "f", "f8", "i8", "u8"):
            return pd.to_numeric(col, errors="coerce")

        return col.astype(out_dtype, errors="ignore")

    def continuous(self, sr: pd.Series) -> bool:
        """Checks if the given series is continuous
        All arithmetic operations can be done on continuous columns
        """
        sr = self.infer_dtype(sr)
        sr_dtype = str(sr.dtype)
        if len(sr.unique()) >= max(min(self.min_num_unique, len(sr)),
                                   self.ctl_mult * np.log(len(sr)))\
           and sr_dtype in ("float64", "int64"):
            return True
        return False

    def categorical(self, sr: pd.Series) -> bool:
        """Checks if the given series is categorical"""
        if pd.api.types.is_categorical_dtype(sr) is True:
            return True

        if not self.continuous(sr):
            return True
        return False

    def ordinal(self, sr: pd.Series) -> bool:
        """Checks if the given series is ordinal
        A series whose values can be ordered is ordinal
        Columns which are categorical in nature, but contain numbers/dates/bool
        are ordinal too. E.g. [2, 1, 1, 2, 7, 2, 1, 7]
        """
        if (pd.api.types.is_categorical_dtype(sr) is True
            and sr.cat.ordered is True)\
           or pd.api.types.is_bool_dtype(sr) is True:
            return True

        sr_inferred = self.infer_dtype(sr)
        if sr_inferred.dtype in ("float64", "int64")\
           or sr_inferred.dtype.kind in ("M", "m"):
            return True
        return False

    def affine(self, sr: pd.Series) -> bool:
        """Checks if the given series is affine
        Continuous columns along with the columns of type DateTime
        and the Timedelta are affine
        """
        if self.continuous(sr) is True\
           or self.infer_dtype(sr).dtype.kind in ("M", "m"):
            return True
        return False

    def same_domain(self, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        """Checks if the given columns have the same domain of values"""
        sr_a = self.infer_dtype(sr_a)
        sr_b = self.infer_dtype(sr_b)
        if self.categorical(sr_a) is True and self.categorical(sr_b) is True\
           and sr_a.dtype.kind != 'M' and sr_b.dtype.kind != 'M'\
           and set(sr_a.dropna().unique()) == set(sr_b.dropna().unique()):
            return True

        return False
