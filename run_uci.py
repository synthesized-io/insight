from datetime import datetime
import sys
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from synthesized.core import BasicSynthesizer
from synthesized.core.classifiers import BasicClassifier


# args
print()
dataset = sys.argv[1]
iterations = int(sys.argv[2])

# load dataset
data = pd.read_csv('data/uci/{}.csv'.format(dataset))
data.sort_index(inplace=True)

# optional args
if len(sys.argv) > 3:
    target = sys.argv[3]
else:
    target = data.columns[-1]

# prepare data
original, heldout = train_test_split(data, test_size=0.2)
original.reset_index(drop=True, inplace=True)
heldout.reset_index(drop=True, inplace=True)

# original data
print(original.head(5))
print()

# initialize synthesizer
synthesizer = BasicSynthesizer(data=data, iterations=iterations)
print(repr(synthesizer))

# original score
train = synthesizer.preprocess(data=original.copy())
estimator = DummyClassifier()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
test = synthesizer.preprocess(data=heldout.copy())
dummy_score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
estimator = LogisticRegression()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
print('original:', score, dummy_score)
print()

# original classifier score
with BasicClassifier(data=data, target_label=target, iterations=iterations) as classifier:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    classifier.learn(data=original.copy(), verbose=10000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    classified = classifier.classify(data=heldout.drop(labels=target, axis=1))
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    print()

score = (classified[target] == heldout[target]).sum() / len(classified)
print('original classifier:', score)
print()

# synthesis
with synthesizer:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    synthesizer.learn(data=original.copy(), verbose=10000)  # filenames=('dataset.tfrecords',)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    synthesized = synthesizer.synthesize(n=10000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    print()

# synthetic data
print(synthesized.head(10))
print()

# synthetic score
train = synthesizer.preprocess(data=synthesized.copy())
estimator = DummyClassifier()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
test = synthesizer.preprocess(data=heldout.copy())
dummy_score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
estimator = LogisticRegression()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
print('synthesized:', score, dummy_score)
print()

# synthetic classifier score
with BasicClassifier(data=data, target_label=target, iterations=iterations) as classifier:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    classifier.learn(data=synthesized.copy(), verbose=10000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    classified = classifier.classify(data=heldout.drop(labels=target, axis=1))
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    print()

score = (classified[target] == heldout[target]).sum() / len(classified)
print('synthesized classifier:', score)
print()
