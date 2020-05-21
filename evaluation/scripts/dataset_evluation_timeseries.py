import os
import json
import warnings
import pandas as pd

warnings.filterwarnings(action='ignore', message='numpy.dtype size changed')
warnings.filterwarnings(action='ignore', message='compiletime version 3.5 of module')

if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
os.chdir(os.path.split(workbookDir)[0])


from synthesized.testing.evaluation import Evaluation
from synthesized.testing import plotting as syn_plot

# evaluation = Evaluation(config_path='configs/evaluation/dataset_evaluation.json', name='james')
evaluation_name = os.environ.get('evaluation_name', 'n/a')
branch = os.environ.get('evaluation_branch', 'n/a')
revision = os.environ.get('evaluation_revision', 'n/a')
evaluation = Evaluation(branch=branch, revision=revision, group="dataset_evaluation_timeseries",
                        metrics_file="../series-exp/metrics-series-exp.jsonl")

config_path = os.environ.get('evaluation_config_path', 'n/a')
with open(config_path, 'r') as f:
    configs = json.load(f)
    config = configs["instances"][evaluation_name]
    evaluation.record_config(evaluation=evaluation_name, config=config)


data = pd.read_csv(evaluation.configs[evaluation_name]['data'])
data = data.drop(evaluation.configs[evaluation_name]['ignore_columns'], axis=1)
data.dropna(inplace=True)
data.head(5)

testing = syn_plot.synthesize_and_plot(data=data, name=evaluation_name, evaluation=evaluation, config=config,
                                       metrics={}, test_data=data, plot_basic=False, plot_losses=True,
                                       plot_distances=True, show_distribution_distances=True,
                                       show_distributions=True, show_correlation_distances=True,
                                       show_correlation_matrix=True, show_emd_distances=True,
                                       show_acf_distances=True, show_transition_distances=True, show_series=True)

evaluation.write_metrics()