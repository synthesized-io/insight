import argparse
from typing import Any, Dict, List, Sequence, Type

import pandas as pd
import yaml

from .unifier_assessor import UnifierAssessor
from ..complex import HighDimSynthesizer
from ..complex.unifier import Unifier
from ..insight.metrics import (CramersV, EarthMoversDistance, KendellTauCorrelation, KolmogorovSmirnovDistance,
                               SpearmanRhoCorrelation, TwoColumnMetric)
from ..metadata.factory import MetaExtractor


def compute_metrics(df_splits: Sequence[pd.DataFrame], df_unified: pd.DataFrame,
                    first_order_metrics: Sequence[TwoColumnMetric], second_order_metrics: Sequence[TwoColumnMetric]):
    assessor = UnifierAssessor(df_splits, df_unified)

    results: Dict[str, Any] = {}

    for metric in first_order_metrics:
        name = metric.name if metric.name is not None else metric.__class__.__name__
        results[name] = assessor.get_first_order_metric_distances(metric)

    for metric in second_order_metrics:
        name = metric.name if metric.name is not None else metric.__class__.__name__
        results[name] = assessor.get_second_order_metric_matrices(metric)

    return results


def evaluate_unifier(unifier_class: Type[Unifier], config_path: str,
                     first_order_metrics: Sequence[TwoColumnMetric] = None,
                     second_order_metrics: Sequence[TwoColumnMetric] = None):
    with open(config_path, "r") as in_f:
        eval_config = yaml.safe_load(in_f)

    if first_order_metrics is None:
        first_order_metrics = (KolmogorovSmirnovDistance(), EarthMoversDistance())
    if second_order_metrics is None:
        second_order_metrics = (CramersV(), KendellTauCorrelation(), SpearmanRhoCorrelation())

    output: Dict[Any, Any] = {}

    for test, test_config in eval_config.items():
        print(f"Starting {test}")
        data_path = f"data/{test_config['data_path']}"
        df = pd.read_csv(data_path)
        output[test] = {}

        print("Splitting DataFrames")
        df_splits: List[pd.DataFrame] = []
        df_meta_splits: List[pd.DataFrame] = []
        for split in test_config["splits"]:
            df_split = df[split].sample(test_config["num_rows"])
            df_splits.append(df_split)
            df_meta_splits.append(MetaExtractor.extract(df_split))

        unifier = unifier_class()
        synthesizers: List[HighDimSynthesizer] = []
        for i, (df, df_meta) in enumerate(zip(df_splits, df_meta_splits)):
            print(f"Updating with DataFrame {i}")
            unifier.update(df, df_meta, num_iterations=test_config["num_iterations"])
            if test_config["compare_w_highdim"]:
                synthesizer = HighDimSynthesizer(df_meta)
                synthesizer.learn(df, num_iterations=test_config["num_iterations"])
                synthesizers.append(synthesizer)
                df_synth = synthesizer.synthesize(num_rows=test_config["num_rows"])
                output[test][f"HighDim_{i}"] = compute_metrics(df_splits=[df], df_unified=df_synth,
                                                               first_order_metrics=first_order_metrics,
                                                               second_order_metrics=second_order_metrics)

        print("Querying")
        df_unified = unifier.query(columns=test_config["query"], num_rows=test_config["num_rows"])
        output[test].update(compute_metrics(df_splits=df_splits, df_unified=df_unified,
                                            first_order_metrics=first_order_metrics,
                                            second_order_metrics=second_order_metrics))

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unifier_name", type=str, help="class name for unifier")
    parser.add_argument("config_path", type=str, help="path to yaml file specifying config of test")
    args = parser.parse_args()

    unifier_class = Unifier.subclasses[args.unifier_name]

    print(evaluate_unifier(unifier_class, args.config_path))
