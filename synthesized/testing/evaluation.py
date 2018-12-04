import collections
import datetime
import json
import os
from collections import OrderedDict

import numpy as np


class Evaluation:
    def __init__(self):
        self.evaluation = os.environ['EVALUATION']
        self.metrics_path = os.environ['METRICS_PATH']
        self.config = json.loads(os.environ['CONFIG'], object_pairs_hook=collections.OrderedDict)
        self.metrics = OrderedDict()

    def __setitem__(self, key, value):
        if key in self.metrics:
            self.metrics[key].append(value)
        else:
            self.metrics[key] = [value]

    def write_metrics(self):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.metrics_path, 'a') as f:
            data = OrderedDict()
            data['evaluation'] = self.evaluation
            data['branch'] = os.environ['BRANCH']
            data['revision'] = os.environ['REVISION']
            data['timestamp'] = timestamp
            data['config'] = json.dumps(self.config)
            for name, vals in self.metrics.items():
                data[name + '_mean'] = np.mean(vals)
                data[name + '_std'] = np.std(vals)
            json.dump(data, f)
            f.write('\n')