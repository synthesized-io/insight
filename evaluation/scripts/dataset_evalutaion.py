import os
import json

if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
os.chdir(os.path.split(os.path.split(workbookDir)[0])[0])

import pandas as pd
from sklearn.model_selection import train_test_split

from synthesized.testing.evaluation import Evaluation, synthesize_and_plot


evaluation_name = os.environ.get('evaluation_name', 'james')
branch = os.environ.get('evaluation_branch', 'test')
revision = os.environ.get('evaluation_revision', 'a1b2c3d')
evaluation = Evaluation(branch=branch, revision=revision, group="dataset_evaluation",
                        metrics_file="../highdim-exp/metrics-highdim-exp.jsonl")

config_path = os.environ.get('evaluation_config_path', 'configs/evaluation/highdim-exp_dataset_evaluation.json')
with open(config_path, 'r') as f:
    configs = json.load(f)
    config = configs["instances"][evaluation_name]
    evaluation.record_config(evaluation=evaluation_name, config=config)

data = pd.read_csv(evaluation.configs[evaluation_name]['data'])
data = data.drop(evaluation.configs[evaluation_name]['ignore_columns'], axis=1)
data.dropna(inplace=True)
data.head(5)


train, test = train_test_split(data, test_size=0.2, random_state=0)
testing = synthesize_and_plot(data=data, name=evaluation_name, evaluation=evaluation, config=config,
                              eval_metrics=[], test_data=test, plot_basic=False, plot_losses=True,
                              plot_distances=True, show_distributions=True,
                              show_distribution_distances=True, show_emd_distances=True,
                              show_correlation_distances=True, show_correlation_matrix=True,
                              show_cramers_v_distances=True, show_cramers_v_matrix=True)

try:
    utility = testing.utility(target=evaluation.configs[evaluation_name]['target'])
except:
    utility = 0.0

evaluation.record_metric(evaluation=evaluation_name, key='utility', value=utility)
evaluation.write_metrics()

