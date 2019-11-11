Synthesized
--------

[![CircleCI](https://circleci.com/gh/synthesized-io/synthesized.svg?style=svg&circle-token=a798b03cdec6651b6604af9121cd5ad12a9c691d)](https://circleci.com/gh/synthesized-io/synthesized)

[Contributing](https://github.com/synthesized-io/synthesized/blob/master/CONTRIBUTING.md)
--------------
Please follow the link to read about contributing guidlines.

Overview
--------

synthesized.io

Installation / Usage
--------------------

Clone the repo:

    $ git clone git@github.com:synthesized-io/synthesized.git
    $ python setup.py install
    
Project setup
-------------

### Create a virtualenv

1. Go to the project directory:

        cd synthesized

2. Install virtualenv command:

        pip3 install virtualenv
        
Note: if you cannot install venv without root check that virtualenv executable is owned by your user

3. Create a virtual environment:

        make venv

4. Run Linter (flake8 and mypy)

        make lint

5. Run Tests (both unit and integration):

        make test
        
6. Run Unit Tests

        make unit-test
        
7. Build a binary wheel package

        make build

8. Run lint, test and build a package

        make
Example
-------

TBD
