import os
import json

if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
os.chdir(os.path.split(os.path.split(workbookDir)[0])[0])

from synthesized.testing.evaluation import Evaluation
from synthesized.testing import synthetic_distributions as syn_dist
from synthesized.testing import plotting as syn_plot

branch = os.environ.get('evaluation_branch', 'n/a')
revision = os.environ.get('evaluation_revision', 'n/a')
group = "synthetic"
config_path = os.environ.get('evaluation_config_path', "configs/evaluation/synthetic_distributions.json")
with open(config_path, 'r') as f:
    configs = json.load(f)
    config = configs["instances"]["synthetic"]
evaluation = Evaluation(branch=branch, revision=revision, group=group,
                        metrics_file='../series-exp/metrics-series-exp.jsonl')

# Time-series

# Continuous
data = syn_dist.create_time_series_data(func=syn_dist.additive_sine(a=10, p=1000, sd=2), length=10000)
metrics = dict(eval_metrics.default_metrics)
metrics["max_acf_distance"] = eval_metrics.max_autocorrelation_distance
metrics["max_pacf_distance"] = eval_metrics.max_partial_autocorrelation_distance
metrics["mean_squared_error"] = eval_metrics.mean_squared_error_closure(col="x", baseline=2**4)
_ = syn_plot.synthesize_and_plot(data, "sine_additive_noise", evaluation=evaluation,
                                 metrics=metrics, config=config, time_series=True,
                                 max_lag=100)

# Categorical
data = syn_dist.create_time_series_data(func=syn_dist.categorical_auto_regressive(n_classes=10, sd=2), length=10000)
metrics = dict(eval_metrics.default_metrics)
metrics["max_auto_association_distance"] = eval_metrics.max_categorical_auto_association_distance
_ = syn_plot.synthesize_and_plot(data, "first_order_markov", evaluation=evaluation,
                                 metrics=metrics, config=config, time_series=True,
                                 max_lag=100)

evaluation.write_metrics()