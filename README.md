Overview
--------

synthesized.io

Installation / Usage
--------------------

Clone the repo:

    $ git clone git@github.com:synthesized-io/synthesized.git
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
        
4. Install deps:

        pip install -r requriements-dev.txt
        
5. Run Tests

        python -m pytest

Note: `pytest` command may run a system version. Call via module forces to run venv version

Example
-------

TBD
