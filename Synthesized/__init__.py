# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# Setup the Synth module

__version__ = '0.1'

try:
    # this var is injected in the setup build to enable
    # the retrieval of the version number without actually
    # importing the un-built submodules.
    __SYNTHESIZED_SETUP__
except NameError:
    __SYNTHESIZED_SETUP__ = False

if __SYNTHESIZED_SETUP__:
    import sys
    import os
    sys.stderr.write('Partial import of Synthesized during the build process.' + os.linesep)
else:
    __all__ = [
        'autoencode',
        'balance',
        'testing'
    ]

    # top-level imports - if any
    # todo


def setup_module(module):
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
