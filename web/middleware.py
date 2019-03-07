import logging
import sys

from flask.json import JSONEncoder


def configure_logger():
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    ))

    root = logging.root
    root.setLevel(logging.INFO)
    root.addHandler(stdout_handler)


# By default NaN is serialized as "NaN". We enforce "null" instead.
class JSONCompliantEncoder(JSONEncoder):
    def __init__(self, *args, **kwargs):
        kwargs["ignore_nan"] = True
        super().__init__(*args, **kwargs)
