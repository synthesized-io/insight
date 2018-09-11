from datetime import datetime
import sys
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from synthesized.core import BasicSynthesizer


# args
print()
dataset = sys.argv[1]
iterations = int(sys.argv[2])

# load dataset
data = pd.read_csv('data/uci/{}.csv'.format(dataset))

# optional args
if len(sys.argv) > 3:
    target = sys.argv[3]
else:
    target = data.columns[-1]

# initialize synthesizer
synthesizer = BasicSynthesizer(data=data, iterations=iterations)
print(repr(synthesizer))

# prepare data
original, test = train_test_split(data, test_size=0.2)
test = synthesizer.preprocess(data=test)

# original data
print(original.head(5))
print()

# original score
train = synthesizer.preprocess(data=original.copy())
estimator = DummyClassifier()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
dummy_score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
estimator = LogisticRegression()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
print('original:', score, dummy_score)
print()

# synthesis
with synthesizer:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    synthesizer.learn(data=original, verbose=10000)  # filenames=('dataset.tfrecords',)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    synthesized = synthesizer.synthesize(n=10000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    print()

# synthetic data
print(synthesized.head(10))
print()

# synthetic score
train = synthesizer.preprocess(data=synthesized)
estimator = DummyClassifier()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
dummy_score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
estimator = LogisticRegression()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
print('synthesized:', score, dummy_score)
print()
