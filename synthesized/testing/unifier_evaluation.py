from typing import Any, Dict, List, Sequence, Type

import pandas as pd
import yaml

from ..complex import HighDimSynthesizer
from ..complex.unifier import Unifier
from ..insight.metrics import (CramersV, EarthMoversDistance, KendallTauCorrelation, KolmogorovSmirnovDistance,
                               SpearmanRhoCorrelation, TwoColumnMetric)
from ..insight.unifier import UnifierAssessor, UnifierModellingAssessor
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


def compute_modelling_metrics(df_splits: Sequence[pd.DataFrame], df_unified: pd.DataFrame, target: str, predictors: Sequence[str]):
    assessor = UnifierModellingAssessor(df_unified, sub_dfs=df_splits)
    models = ['Linear', 'GradientBoosting', 'RandomForest', 'MLP', 'LinearSVM']
    results = {}

    for model in models:
        results[model] = assessor.get_metric_score_for_unified_df(target=target, predictors=predictors, model=model)

    return results


def evaluate_unifier(unifier_class: Type[Unifier], config_path: str,
                     first_order_metrics: Sequence[TwoColumnMetric] = None,
                     second_order_metrics: Sequence[TwoColumnMetric] = None):
    with open(config_path, "r") as in_f:
        eval_config = yaml.safe_load(in_f)

    if first_order_metrics is None:
        first_order_metrics = (KolmogorovSmirnovDistance(), EarthMoversDistance())
    if second_order_metrics is None:
        second_order_metrics = (CramersV(), KendallTauCorrelation(), SpearmanRhoCorrelation())

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
            unifier.update(dfs=df, df_metas=df_meta, num_iterations=test_config["num_iterations"])
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

        if "query" in test_config and "predictors" in test_config:
            output[test].update(compute_modelling_metrics(df_splits, df_unified, target=test_config["target"],
                                                          predictors=test_config["predictors"]))

    return output
