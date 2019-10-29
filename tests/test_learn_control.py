import os
from datetime import datetime

import pandas as pd

from synthesized import HighDimSynthesizer


files_dir = '/Users/tonbadal/PycharmProjects/synthesized-web/project_templates/templates/'
files = os.listdir(files_dir)
for file in files:
    print("\nDATASET '{}'\n".format(file))
    data = pd.read_csv(files_dir + file)

    t = datetime.now()
    with HighDimSynthesizer(df=data
                            ) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=10000)
        synthesized = synthesizer.synthesize(num_rows=len(data.dropna()))
    print('Total time: ', datetime.now() - t)
    print('\n================================================\n')
