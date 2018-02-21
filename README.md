[![Build status](https://travis-ci.org/tgsmith61591/smrt.svg?branch=master)](https://travis-ci.org/tgsmith61591/smrt)
[![codecov](https://codecov.io/gh/tgsmith61591/smrt/branch/master/graph/badge.svg)](https://codecov.io/gh/tgsmith61591/smrt)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg)
![Supported versions](https://img.shields.io/badge/python-3.5-blue.svg)


# Constraints-preserving data pipelines 


## Installation

Installation is easy. After cloning the project onto your machine and installing the required dependencies,
simply use the `setup.py` file:

```bash
$ git clone ...
$ cd Synth
$ python setup.py install
```

## About

...

__See [the paper](doc/smrt.tex) for more in-depth reference.__

## Example

The [SMRT example](examples/) is an ipython notebook with reproducible code and data that compares an imbalanced
variant of the MNIST dataset after being balanced with both SMOTE and SMRT. The following are several of the resulting
images produced from both SMOTE and SMRT, respectively. Even visually, it's evident that SMRT better synthesizes data
that resembles the input data.

### Original:

The MNIST dataset was amended to contain only zeros and ones in an unbalanced (~1:100, respectively) ratio. The top row
are the original MNIST images, the second row is the SMRT-generated images, and the bottom row is the SMOTE-generated
images:
<br/>
<img src="examples/img/mnist_smrt_smote.png" width="600" alt="Original"/>

### Notes

- See [examples](examples/) for usage
- See [the paper](doc/smrt.tex) for more in-depth documentation
- Information on [the authors](AUTHORS.md)
