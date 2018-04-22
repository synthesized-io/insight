synthesized
===============================

version number: 1.0.0
author: Synthesized Ltd.

Overview
--------

synthesized.io

Installation / Usage
--------------------

Clone the repo:

    $ git clone https://github.com/nbaldin/synthesized.git
    $ python setup.py install
    
Contributing
------------

### Create a virtualenv

1. Go to the project directory:

        cd synthesized

1. Install virtualenv command:

        pip3 install --user virtualenv
        
Note: if you cannot install venv without root check that virtualenv executable is owned by your user

2. Create a virtual environment:

        virtualenv -p python3 venv
        
3. Active it:

        source venv/bin/activate

4. Install requirements:

        pip install -r requirements.txt

5. Install the current venv as a kernel for jupyter:

        python -m ipykernel install --user --name synthesized
        
Run Tests
--------

        python -m nose

Note: `nosetests` command may run a system version. Call via module forces to run venv version

Example
-------

TBD