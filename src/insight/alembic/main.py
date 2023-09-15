# Python script that will apply the migrations up to head
import os
import sys

import alembic.config

here = os.path.dirname(os.path.abspath(__file__))

args = sys.argv[1:]
args = args if len(args) > 0 else ["upgrade", "head"] # default
alembic_args = [
    '-c', os.path.join(here, 'alembic.ini'),
] + args


def main():
    alembic.config.main(argv=alembic_args)
