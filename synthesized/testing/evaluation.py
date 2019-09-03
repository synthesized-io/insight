import datetime
import json
from collections import OrderedDict

import numpy as np


class Evaluation:
    def __init__(self, branch, revision, group, metrics_file='../metrics.jsonl'):
        self.branch = branch
        self.revision = revision
        self.group = group
        self.metrics_file = metrics_file
        self.metrics = OrderedDict()
        self.configs = OrderedDict()

    def record_config(self, evaluation, config):
        self.configs[evaluation] = config

    def record_metric(self, evaluation, key, value):
        if evaluation not in self.metrics:
            self.metrics[evaluation] = OrderedDict()
        evaluation_metrics = self.metrics[evaluation]
        if key in evaluation_metrics:
            evaluation_metrics[key].append(value)
        else:
            evaluation_metrics[key] = [value]

    def write_metrics(self):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.metrics_file, 'a') as f:
            for evaluation in self.metrics.keys():
                data = OrderedDict()
                data['evaluation'] = evaluation
                data['branch'] = self.branch
                data['revision'] = self.revision
                data['timestamp'] = timestamp
                if evaluation in self.configs:
                    data['config'] = json.dumps(self.configs[evaluation])
                for name, vals in self.metrics.items():
                    data[name + '_mean'] = np.mean(vals)
                    data[name + '_std'] = np.std(vals)
                    data[name + '_count'] = len(vals)
                json.dump(data, f)
                f.write('\n')
