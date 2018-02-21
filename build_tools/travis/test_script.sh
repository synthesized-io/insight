#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import multiprocessing as mp; print('%d CPUs' % mp.cpu_count())"

run_tests() {
    # Get into a temp directory to run test from the installed Synth and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR
    # We need the setup.cfg for the nose settings
    cp setup.cfg $TEST_DIR
    cd $TEST_DIR

    if [[ "$COVERAGE" == "true" ]]; then
        nosetests -s --with-coverage --with-timer --timer-top-n 20 Synth
        codecov --token=6de72173-c50f-4ebb-b7f1-22d592f56b01
    else
        nosetests -s --with-timer --timer-top-n 20 Synth
    fi

}

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi