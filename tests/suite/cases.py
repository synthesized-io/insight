import glob
import os

import pandas as pd

from synthesized.suite.selector import CSVDirectorySelector


def _compose_case_from_directory(case_directory):
    csv_directory = os.path.join(case_directory, "data")
    assert os.path.isdir(csv_directory), csv_directory
    csv_filename_list = glob.glob(os.path.join(csv_directory, "*.csv"))

    selector = CSVDirectorySelector(directory_path=csv_directory)
    assert len(selector.data) == len(csv_filename_list)
    assert sorted(selector.data.keys()) == sorted(
        map(lambda x: os.path.basename(x).replace(".csv", ""), csv_filename_list)
    )
    for table_name in selector.data.keys():
        df = pd.read_csv(os.path.join(csv_directory, f"{table_name}.csv"))
        assert len(selector.data[table_name]) == len(df)

    rule_stmt_df = pd.read_csv(os.path.join(case_directory, "functional_mappings.csv"))
    join_stmt_df = pd.read_csv(os.path.join(case_directory, "join_statements.csv"))
    return selector, rule_stmt_df, join_stmt_df


def rules_annotated():
    return "tests/suite/assets/rules_annotated/"


def rules_raw():
    return [
        stmt
        for _, (_, stmt) in pd.read_csv(
            "tests/suite/assets/rules_raw.csv"
        ).iterrows()
    ]


def case_tpch():
    return _compose_case_from_directory(
        case_directory="tests/suite/assets/cases/tpch"
    )


def case_synthetic():
    return _compose_case_from_directory(
        case_directory="tests/suite/assets/cases/synthetic"
    )


def case_nn():
    fm_file = "tests/suite/assets/cases/nn-suite/sql_statements.csv"
    join_file = "tests/suite/assets/cases/nn-suite/join_statement-v2.csv"
    data_path = "tests/suite/assets/cases/nn-suite/data"

    df_fm = pd.read_csv(fm_file)
    df_join = pd.read_csv(join_file)[["TARGET", "SQL_JOIN_STATEMENT"]]
    selector = CSVDirectorySelector(data_path, sep="|")

    return selector, df_fm, df_join
