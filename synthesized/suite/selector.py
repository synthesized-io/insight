import abc
import os
from typing import Dict, NewType, Optional

import pandas as pd
import pandasql as ps

from .parsers.join import ParsedJoinStmt

SelectedData = NewType("SelectedData", Dict[str, pd.DataFrame])


def selected_data_to_dict(data: SelectedData) -> Dict[str, dict]:
    return {key: df.to_dict(orient="list") for key, df in data.items()}


class Selector(abc.ABC):
    """
    An instance of the class represents immutable data storage select according to
        provided schema.

    """

    data: SelectedData = SelectedData({})

    def join(self, stmt: ParsedJoinStmt, force_lowercase: bool = True) -> pd.DataFrame:
        """
        Joins the `self.data` according to the given `stmt`.

        The resulting df columns are in format `table_name.column_name`.
         If `force_lowercase` is true -- in lowercase.
        """
        affected_table_names = stmt.table_list
        raw_join_stmt = stmt.raw
        # compose sql statement by parts
        select_columns = ", ".join(
            [
                f"[{table_name.lower() if force_lowercase else table_name}].*"
                for table_name in affected_table_names
            ]
        )
        sql_stmt = f"select {select_columns} {raw_join_stmt}"
        df = self.execute_sql_stmt(sql_stmt)
        # alter the columns' names to match {table}.{column}
        df.columns = sum(
            [
                [
                    f"{table_name}.{column_name}"
                    for column_name in self.data[table_name].columns
                ]
                for table_name in affected_table_names
            ],
            [],
        )
        if force_lowercase:
            df.columns = [col.lower() for col in df.columns]

        if len(df) == 0:
            raise ValueError("Joining data returns empty dataframe")
        return df

    def unjoin(self, df: pd.DataFrame, stmt: ParsedJoinStmt) -> SelectedData:
        """
        Reverts the join `stmt` resulted in the given `df`.

        If table's column is not present in `df`, but present in `self.schema`,
            it will be dropped.
        """
        affected_table_names = stmt.table_list
        selected_data = SelectedData({})
        for table_name in affected_table_names:
            # find the required columns from table name references
            related_column_names = [
                joined_column_name
                for joined_column_name in df.columns
                if joined_column_name.startswith(f"{table_name}.")
            ]
            selected_data[table_name] = (
                df[related_column_names].drop_duplicates().reset_index(drop=True)
            )
            # remove table name reference from columns
            selected_data[table_name].columns = [
                joined_column_name.replace(f"{table_name}.", "")
                for joined_column_name in selected_data[table_name].columns
            ]
        return selected_data

    @abc.abstractmethod
    def execute_sql_stmt(self, sql_stmt: str) -> pd.DataFrame:
        pass


class PandaSQLSelector(Selector):
    def execute_sql_stmt(self, sql_stmt: str) -> pd.DataFrame:
        return ps.sqldf(sql_stmt, self.data)


class CSVDirectorySelector(PandaSQLSelector):
    """
    Names of the tables should precisely match the csv filenames.
    """

    directory_path: str
    sep: str

    def __init__(self, directory_path: str, sep: Optional[str] = None):
        assert os.path.isdir(directory_path), directory_path
        self.directory_path = directory_path
        self.sep = sep if sep else ","

        self.data = self.read_data_directory()

    def read_data_directory(self):
        result = SelectedData({})

        for dirpath, _, filenames in os.walk(self.directory_path):
            for filename in filenames:
                if filename.endswith(".csv"):
                    table_name = f"{filename[:-4].lower()}"
                    result[table_name] = pd.read_csv(
                        os.path.join(dirpath, filename), sep=self.sep
                    ).rename(columns=lambda c: c.lower())

        return result
