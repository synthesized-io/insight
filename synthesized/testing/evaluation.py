import collections
import datetime
import json
import os
from collections import OrderedDict

import numpy as np

EVALUATION_NAME = 'EVALUATION_NAME'
EVALUATION_CONFIG_PATH = 'EVALUATION_CONFIG_PATH'
EVALUATION_BRANCH = 'EVALUATION_BRANCH'
EVALUATION_REVISION = 'EVALUATION_REVISION'


class Evaluation:
    def __init__(self, config_path=None, name=None, metrics_file='../metrics.jsonl'):
        if name:
            self.name = name
        else:
            self.name = os.environ[EVALUATION_NAME]

        if config_path:
            self.config_path = config_path
        else:
            self.config_path = os.environ[EVALUATION_CONFIG_PATH]

        self.metrics_file = metrics_file

        with open(self.config_path, 'r') as f:
            configs = json.load(f, object_pairs_hook=collections.OrderedDict)
            self.config = configs['instances'][name]
        self.metrics = OrderedDict()

    def __setitem__(self, key, value):
        if key in self.metrics:
            self.metrics[key].append(value)
        else:
            self.metrics[key] = [value]

    def write_metrics(self):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.metrics_file, 'a') as f:
            data = OrderedDict()
            data['evaluation'] = self.name
            data['branch'] = os.environ.get(EVALUATION_BRANCH, 'n/a')
            data['revision'] = os.environ.get(EVALUATION_REVISION, 'n/a')
            data['timestamp'] = timestamp
            data['config'] = json.dumps(self.config)
            for name, vals in self.metrics.items():
                data[name + '_mean'] = np.mean(vals)
                data[name + '_std'] = np.std(vals)
                data[name + '_count'] = len(vals)
            json.dump(data, f)
            f.write('\n')
