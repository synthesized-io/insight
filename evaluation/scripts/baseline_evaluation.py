import os
import json

if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
os.chdir(os.path.split(workbookDir)[0])

import pandas as pd

from synthesized.testing.evaluation import Evaluation, baseline_evaluation_and_plot
from IPython.display import Markdown, display

evaluation_name = os.environ.get('evaluation_name', 'n/a')
branch = 'master'
revision = os.environ.get('evaluation_revision', 'n/a')
server_mode = os.environ.get('SERVER_MODE')
metrics_file = f"../{server_mode}/metrics_baseline-{server_mode}.jsonl"

evaluation = Evaluation(branch=branch, revision=revision, group='baseline_evaluation',
                        metrics_file=metrics_file)

config_path = os.environ.get('baseline_evaluation_config_path', 'n/a')

with open(config_path, 'r') as f:
    configs = json.load(f)

for evaluation_name, config in configs['instances'].items():
    display(Markdown("## {}".format(evaluation_name)))

    evaluation.record_config(evaluation=evaluation_name, config=config)
    data = pd.read_csv(evaluation.configs[evaluation_name]['data'])
    data = data.drop(evaluation.configs[evaluation_name]['ignore_columns'], axis=1)

    baseline_evaluation_and_plot(data, evaluation_name, evaluation)

evaluation.write_metrics(append=False)
