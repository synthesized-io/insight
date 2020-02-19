import os
import datetime
import simplejson
from collections import OrderedDict

import numpy as np


class Evaluation:
    def __init__(self, branch: str, revision: str, group: str, name: str = 'metrics'):

        self.branch = branch
        self.revision = revision
        self.group = group

        server_mode = os.environ.get('SERVER_MODE')
        assert len(name) > 0
        if server_mode:
            self.metrics_file: str = '../history/metrics/{}-{}.jsonl'.format(name, server_mode)
        else:
            self.metrics_file = '../history/metrics/{}.jsonl'.format(name)

        self.metrics: OrderedDict = OrderedDict()
        self.configs: OrderedDict = OrderedDict()

    def record_config(self, evaluation: str, config: dict):
        self.configs[evaluation] = config

    def record_metric(self, evaluation: str, key: str, value: float):
        if evaluation not in self.metrics:
            self.metrics[evaluation] = OrderedDict()
        evaluation_metrics = self.metrics[evaluation]
        if key in evaluation_metrics:
            evaluation_metrics[key].append(value)
        else:
            evaluation_metrics[key] = [value]

    def write_metrics(self, append=True):
        timestamp = datetime.datetime.now().isoformat()
        write_mode = 'a' if append else 'w'
        with open(self.metrics_file, write_mode) as f:
            for evaluation, evaluation_metrics in self.metrics.items():
                data = OrderedDict()
                data['group'] = self.group
                data['evaluation'] = evaluation
                data['branch'] = self.branch
                data['revision'] = self.revision
                data['timestamp'] = timestamp
                if evaluation in self.configs:
                    data['config'] = simplejson.dumps(self.configs[evaluation], separators=(',', ':'), ignore_nan=True)
                for name, vals in evaluation_metrics.items():
                    data[name + '_mean'] = np.mean(vals)
                    data[name + '_std'] = np.std(vals)
                    data[name + '_count'] = len(vals)
                simplejson.dump(data, f, separators=(',', ':'), ignore_nan=True)
                f.write('\n')
