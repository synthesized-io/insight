import argparse
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from synthesized.core import BasicSynthesizer


print()
print('Parse arguments...')
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help="dataset name")
parser.add_argument('-t', '--target', default=-1, help="target column")
parser.add_argument('--drop', type=str, default=None, help="values to drop")
parser.add_argument('--drop-nans', action='store_true', help="drop nans")
parser.add_argument('-i', '--identifier-label', default=None, help="identifier label")
parser.add_argument('-l', '--lstm-mode', type=int, default=0, help="lstm mode")
parser.add_argument('-n', '--num-iterations', type=int, default=100, help="training iterations")
parser.add_argument(
    '-y', '--hyperparameters', default=None, help="list of hyperparameters (comma, equal)"
)
parser.add_argument('-b', '--tensorboard', action='store_true', help="TensorBoard summaries")
parser.add_argument('--tfrecords', action='store_true', help="from TensorFlow records")
args = parser.parse_args()


print('Load dataset...')
data = pd.read_csv('data/{}.csv'.format(args.dataset))
tfrecords_filename = 'data/{}.tfrecords'.format(args.dataset)
if args.drop is not None:
    data = data.drop(columns=args.drop.split(','))
num_with_nan = len(data)
if args.drop_nans:
    data = data.dropna()
print('Nans dropped:', num_with_nan - len(data), 'of', num_with_nan)
print()


print('Original data...')
print(data.head(5))
print()


print('Initialize synthesizer...')
if args.hyperparameters is None:
    synthesizer = BasicSynthesizer(
        data=data, exclude_encoding_loss=True, summarizer=args.tensorboard,
        lstm_mode=args.lstm_mode, identifier_label=args.identifier_label
    )
else:
    kwargs = [kv.split('=') for kv in args.hyperparameters.split(',')]
    kwargs = {key: float(value) if '.' in value else int(value) for key, value in kwargs}
    synthesizer = BasicSynthesizer(
        data=data, exclude_encoding_loss=True, summarizer=args.tensorboard,
        lstm_mode=args.lstm_mode, identifier_label=args.identifier_label, **kwargs
    )
print(repr(synthesizer))
print()


print('Synthesis...')
with synthesizer:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    if args.tfrecords:
        synthesizer.learn(num_iterations=args.num_iterations, filenames=(tfrecords_filename,))
    else:
        synthesizer.learn(num_iterations=args.num_iterations, data=data.copy())
    synthesized = synthesizer.synthesize(n=1000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
print()

print('Synthesized data...')
print(synthesized.head(5))
print()
