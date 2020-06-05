import os
import json
import logging

if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
# os.chdir(os.path.split(workbookDir)[0])
os.chdir(os.path.split(os.path.split(workbookDir)[0])[0])

import pandas as pd

from synthesized.testing.evaluation import Evaluation, synthesize_and_plot_time_series

logger = logging.getLogger("synthesized")
logger.setLevel("DEBUG")

evaluation_name = os.environ.get('evaluation_name', 'standard_and_poors500_dss')
branch = os.environ.get('evaluation_branch', 'master')
revision = os.environ.get('evaluation_revision', 'a1b2c3d')
evaluation = Evaluation(branch=branch, revision=revision, group="dataset_evaluation_timeseries",
                        metrics_file="../series-exp/metrics-series-exp.jsonl")

# config_path = os.environ.get('evaluation_config_path', 'configs/evaluation/series-exp_dataset_evaluation.json')
config_path = 'configs/evaluation/series-exp_dataset_evaluation.json'
print(config_path)
with open(config_path, 'r') as f:
    configs = json.load(f)
    config = configs["instances"][evaluation_name]
    evaluation.record_config(evaluation=evaluation_name, config=config)


data = pd.read_csv(evaluation.configs[evaluation_name]['data'])
data = data.drop(evaluation.configs[evaluation_name]['ignore_columns'], axis=1)
data.dropna(inplace=True)

data['date'] = pd.to_datetime(data['date'])
test_data = data[data['date'] > '2017-01-01']
data = data[data['date'] <= '2017-01-01']


testing = synthesize_and_plot_time_series(
    data=data, name=evaluation_name, evaluation=evaluation, config=config,
    eval_metrics=[], test_data=test_data,
    plot_basic=False,
    # plot_losses=True,
    # plot_distances=True,
    # show_distribution_distances=True,
    # show_distributions=True,
    # show_correlation_distances=True,
    # show_correlation_matrix=True,
    # show_emd_distances=True,
    show_acf=True,
    # show_transition_distances=True,
    show_series=True
)

evaluation.write_metrics()
