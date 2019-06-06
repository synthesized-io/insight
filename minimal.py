import argparse
from datetime import datetime
import os

import pandas as pd

from synthesized.core import BasicSynthesizer


print()
print(datetime.now().strftime('%H:%M:%S'), 'Parse arguments...', flush=True)
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help="dataset name")
parser.add_argument('-t', '--target', default=-1, help="target column")
parser.add_argument('--drop', type=str, default=None, help="values to drop")
parser.add_argument('--drop-nans', action='store_true', help="drop nans")
parser.add_argument('-i', '--identifier-label', default=None, help="identifier label")
parser.add_argument('-n', '--num-iterations', type=int, default=100, help="training iterations")
parser.add_argument(
    '-y', '--hyperparameters', default='capacity=8,batch_size=8', help="list of hyperparameters (comma, equal)"
)
parser.add_argument('-b', '--tensorboard', action='store_true', help="TensorBoard summaries")
parser.add_argument('--tfrecords', action='store_true', help="from TensorFlow records")
args = parser.parse_args()


print(datetime.now().strftime('%H:%M:%S'), 'Load dataset...', flush=True)
if os.path.isfile(args.dataset):
    data = pd.read_csv(args.dataset)
elif os.path.isfile(os.path.join('data', args.dataset)):
    data = pd.read_csv(os.path.join('data', args.dataset))
elif os.path.isfile(os.path.join('data', args.dataset + '.csv')):
    data = pd.read_csv(os.path.join('data', args.dataset + '.csv'))
else:
    assert False
tfrecords_filename = 'data/{}.tfrecords'.format(args.dataset)
if args.drop is not None:
    data = data.drop(columns=args.drop.split(','))
num_with_nan = len(data)
if args.drop_nans:
    data = data.dropna()
print('Nans dropped:', num_with_nan - len(data), 'of', num_with_nan)
print()


# from random import randrange
# print(len(data.columns))
# for _ in range(len(data.columns) // 2):
#     data = data.drop(data.columns[randrange(len(data.columns))], axis=1)
# print(len(data.columns))


print(datetime.now().strftime('%H:%M:%S'), 'Original data...', flush=True)
print(data.head(5))
print()


print(datetime.now().strftime('%H:%M:%S'), 'Initialize synthesizer...', flush=True)
synthesizer_cls = BasicSynthesizer
if args.hyperparameters is None:
    synthesizer = synthesizer_cls(
        data=data, exclude_encoding_loss=True, summarizer=args.tensorboard,
        identifier_label=args.identifier_label
    )
else:
    assert all('=' in kv or kv == '' for kv in args.hyperparameters.split(','))
    kwargs = [kv.split('=') for kv in args.hyperparameters.split(',') if kv != '']
    kwargs = {key: float(value) if '.' in value else int(value) for key, value in kwargs}
    synthesizer = synthesizer_cls(
        data=data, exclude_encoding_loss=True, summarizer=args.tensorboard,
        identifier_label=args.identifier_label, **kwargs
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
    if args.tfrecords:
        synthesizer.learn(num_iterations=args.num_iterations, filenames=(tfrecords_filename,))
    else:
        synthesizer.learn(num_iterations=args.num_iterations, data=data.copy())
    print(datetime.now().strftime('%H:%M:%S'), 'Finished learning...', flush=True)
    synthesized = synthesizer.synthesize(n=100)
print()

print(datetime.now().strftime('%H:%M:%S'), 'Synthesized data...', flush=True)
print(synthesized.head(5))
print()
