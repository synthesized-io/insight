import os
from datetime import datetime
import json

import pandas as pd

from synthesized import HighDimSynthesizer


files_dir = '../synthesized-web/project_templates/templates/'
# files = os.listdir(files_dir)

j = json.load(open('../synthesized-web/project_templates/meta.json', 'r'))
files = [l["file"].split('/')[1] for l in j["templates"]]

for file in files:
    print("\nDATASET '{}'\n".format(file))
    data = pd.read_csv(files_dir + file)

    t = datetime.now()
    with HighDimSynthesizer(df=data
                            ) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=10000, ds_name=file)
        synthesized = synthesizer.synthesize(num_rows=len(data.dropna()))
    print('Total time: ', datetime.now() - t)
    print('\n================================================\n')
