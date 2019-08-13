import argparse
from datetime import datetime
import os

import pandas as pd

from synthesized import HighDimSynthesizer


print()
print(datetime.now().strftime('%H:%M:%S'), 'Parse arguments...', flush=True)
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='unittest', help="dataset")
parser.add_argument('-i', '--identifier-label', default=None, help="identifier label")
parser.add_argument('-n', '--num-iterations', type=int, default=100, help="training iterations")
parser.add_argument(
    '-y', '--hyperparameters', default='capacity=8,depth=1,batch_size=8',
    help="list of hyperparameters (comma, equal)"
)
parser.add_argument('-b', '--tensorboard', type=str, default=None, help="TensorBoard summaries")
args = parser.parse_args()
print()


print(datetime.now().strftime('%H:%M:%S'), 'Load dataset...', flush=True)
if os.path.isfile(args.dataset):
    filename = args.dataset
elif os.path.isfile(os.path.join('data', args.dataset)):
    filename = os.path.join('data', args.dataset)
elif os.path.isfile(os.path.join('data', args.dataset + '.csv')):
    filename = os.path.join('data', args.dataset + '.csv')
else:
    assert False
df_original = pd.read_csv(filename)
print()


# from random import randrange
# print(len(data.columns))
# for _ in range(len(data.columns) // 2):
#     data = data.drop(data.columns[randrange(len(data.columns))], axis=1)
# print(len(data.columns))


print(datetime.now().strftime('%H:%M:%S'), 'Original data...', flush=True)
print(df_original.head(5))
print()


print(datetime.now().strftime('%H:%M:%S'), 'Initialize synthesizer...', flush=True)
if args.hyperparameters is None:
    synthesizer = HighDimSynthesizer(
        df=df_original, summarizer=args.tensorboard, identifier_label=args.identifier_label
    )
else:
    assert all('=' in kv or kv == '' for kv in args.hyperparameters.split(','))
    kwargs = [kv.split('=') for kv in args.hyperparameters.split(',') if kv != '']
    kwargs = {key: float(value) if '.' in value else int(value) for key, value in kwargs}
    synthesizer = HighDimSynthesizer(
        df=df_original, summarizer=args.tensorboard, identifier_label=args.identifier_label,
        **kwargs
    )
print(repr(synthesizer))
print()


print(datetime.now().strftime('%H:%M:%S'), 'Value types...', flush=True)
for value in synthesizer.values:
    print(value.name, value)
print()


print(datetime.now().strftime('%H:%M:%S'), 'Synthesis...', flush=True)
with synthesizer:
    print(datetime.now().strftime('%H:%M:%S'), 'Start learning...', flush=True)
    synthesizer.learn(num_iterations=args.num_iterations, df_train=df_original, callback_freq=20)
    print(datetime.now().strftime('%H:%M:%S'), 'Finished learning...', flush=True)
    df_synthesized = synthesizer.synthesize(num_rows=10000)
    assert len(df_synthesized) == 10000
print()


print(datetime.now().strftime('%H:%M:%S'), 'Synthesized data...', flush=True)
print(df_synthesized.head(5))
print()
