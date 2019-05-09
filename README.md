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

        pip3 install --user virtualenv
        
Note: if you cannot install venv without root check that virtualenv executable is owned by your user

3. Create a virtual environment:

        virtualenv -p python3 venv
        
4. Active it:

        source venv/bin/activate
        
5. Install deps:

        pip install -r requriements-dev.txt

6. If running on mac OS X, set matplotlib backend:        
        
        mkdir -p ~/.matplotlib
        echo 'backend: TkAgg' > ~/.matplotlib/matplotlibrc

7. Run Tests:

        python -m pytest

Note: `pytest` command may run a system version. Call via module forces to run venv version

Example
-------

TBD
